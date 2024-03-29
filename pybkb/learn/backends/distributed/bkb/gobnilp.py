import ray
import os
import numpy as np
import itertools
import logging
import contextlib
import time
import tqdm
import numba as nb
from operator import itemgetter
from pygobnilp.gobnilp import Gobnilp
from gurobipy import GRB
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from multiprocessing import Pool

from pybkb.utils.probability import *
from pybkb.utils.mp import MPLogger
from pybkb.scores import *
from pybkb.learn.report import LearningReport
from pybkb.bkb import BKB
from pybkb.learn.backends.bkb.gobnilp import BKBGobnilpBackend


class BKBGobnilpDistributedBackend(BKBGobnilpBackend):
    def __init__(
            self,
            score:str,
            palim:int=None,
            only:str=None,
            ray_address:str=None,
            strategy:str='anytime',
            cluster_allocation:float=0.25,
            save_dir:str=None,
            ) -> None:
        """ BKB Gobnilp DAG learning backend that distributes learning over a ray cluster

        Args:
            :param score: Name of scoring function. Choices: mdl_mi, mdl_ent.
            :type score: str
            :param num_learners: Number of ray learner workers to use on each node.
            :type num_learners: int 
            :param num_cluster_nodes: Number of ray nodes that are in the cluster.
            :type num_learners: int 

        Kwargs:
            :param palim: Limit on the number of parent sets. 
            :type palim: int
            :param only: Return only the data score or model score or both. Options: data, model, None. Defaults to None which means both.
            :type only: str
            :param ray_address: The address of the Ray cluster. Defaults to auto.
            :type ray_address: str
            :param strategy: A calculation strategy to be used during distrbuted learning, options are: precompute and anytime.
                Precompute will calculate all necessary scores/joint probabilities before DAGs. Anytime will calculate everything for
                each example in a distributed fashion, i.e. can't leverage reusing scores and joint hashtables.
            :type strategy: str
            :param cluster_allocation: Percentage of cluster nodes to allocate. Number between 0 and 1. 
            :type cluster_allocation: float
            :param save_dir: A directory where to save learned BKFs.
            :type save_dir: str
        """
        if ray_address is None:
            print('Warning: You did not pass a ray address so assuming ray has already been initialized.')
        self.ray_address = ray_address
        self.cluster_allocation = cluster_allocation
        self.strategy = strategy
        self.save_dir = save_dir
        # Need to None this attribute out, should eventually refactor.
        self.score_pool = None
        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        super().__init__(score, palim, only)

    def setup_cluster(self):
        """ Function to initialize the ray cluster and bundle workers onto each cluster node to
        optimize for space.
        """
        # Initialize
        if self.ray_address is not None:
            ray.init(address=self.ray_address)
        ## Create bundles
        # Number of cpus to allocate based on cluster utilization
        self.available_cpus = ray.available_resources()["CPU"]
        self.num_cpus = int(self.cluster_allocation * self.available_cpus)
        self.num_nodes = len(ray.nodes())
        self.available_cpus_per_node = int(self.available_cpus // self.num_nodes)
        self.num_workers_per_node = self.num_cpus // self.num_nodes
        self.num_score_workers_per_worker = int((self.available_cpus_per_node - self.num_workers_per_node - 2) // self.num_workers_per_node)
        if self.num_score_workers_per_worker <= 1:
            self.num_score_workers_per_worker = None
        bundles = [{"CPU": self.num_workers_per_node} for _ in range(self.num_nodes)]
        ## Create Placement Group
        pg = placement_group(bundles, strategy='STRICT_SPREAD')
        ## Wait for the placement group to be ready
        ray.get(pg.ready())
        return pg, bundles
    
    def put_data_on_cluster(self, data, feature_states, srcs, feature_states_index_map, scores, store):
        """ Will put all sharable data into the ray object store.
        """
        self.data_id = ray.put(data)
        self.data_eff_id = ray.put(BKBGobnilpBackend.format_data_for_probs(data, pickleable=True))
        self.feature_states_id = ray.put(np.array(feature_states))
        self.feature_states_index_map_id = ray.put(feature_states_index_map)
        self.srcs_id = ray.put(srcs)
        self.scores_id = ray.put(scores)
        self.store_id = ray.put(store)
        return 
    
    def split_data_over_cluster(self, data, logger=None):
        clusters = np.array_split(np.array([i for i in range(data.shape[0])]), self.num_cpus)
        return clusters
    
    def learn(
            self, 
            data, 
            feature_states,
            srcs,
            verbose:bool=False, 
            end_stage:str=None,
            begin_stage:str=None,
            scores:dict=None, 
            store:dict=None,
            nsols=1,
            kbest=True,
            mec=False,
            pruning=False,
            ):
        """ Learns the best set of BKFs from the data.

        Args:
            :param data: Full database to learn over.
            :type data: np.array
            :param feature_states: List of feature instantiations.
            :type feature_states: list
            :param srcs: A list of source names for each data instance.
            :type srcs: list

        Kwargs:
            :param verbose: Whether to print progress bars.
            :type verbose: bool
            :param scores: Pre-calculated scores dictionary.
            :type scores: dict
            :param store: Pre-calculated store of joint probabilities.
            :type store: dict
            :param end_stage: Stage to end learning. Passing 'scores' will end learning after all
                scores are calculated.
            :type end_stage: str
            :param begin_stage: Stage to begin learning. Passing 'scores' will assume a scores file has been passed and
                will proceed directly to BKF learning.
            :type begin_stage: str
            :param nsols: Number of BKBs to learn per example.
            :type nsols: int
            :param kbest: Whether the nsols learned BKBs should be a highest scoring set of nsols BKBs.
            :type kbest: bool
            :param mec: Whether only one BKB per Markov equivalence class should be feasible.
            :type mec: bool
            :param pruning: Whether not to include parent sets which cannot be optimal when acyclicity is the only constraint.
            :type pruning: bool
        """
        # Set some additional class attributes
        self.nsols = nsols
        self.kbest = kbest
        self.mec = mec
        self.pruning = pruning
        # Run learning
        if self.strategy == 'precompute':
            return self.learn_precompute(data, feature_states, srcs, verbose, begin_stage, end_stage, scores, store)
        elif self.strategy == 'anytime':
            return self.learn_anytime(data, feature_states, srcs, verbose, begin_stage, end_stage, scores, store)
        else:
            raise ValueError(f'Unknown strategy: {self.strategy}')
    
    
    ### Anytime learning strategy functions

    def setup_anytime_work(self, data, logger, verbose, pg, bundles, begin_stage, end_stage):
        row_splits = self.split_data_over_cluster(data, logger)
        # Split row indices over number of available CPUs
        res_ids = []
        split_counter = 0
        bundle_idx = 0
        for split_id, split in tqdm.tqdm(enumerate(row_splits), desc='Setting up remote calls', disable=not verbose, leave=False, total=len(row_splits)):
            split_counter += 1
            if split_counter > self.num_workers_per_node:
                # Move on to the next bundle
                split_counter = 0
                bundle_idx += 1
            # Build BKF
            return_id = learn_distributed.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=bundle_idx,
                        )
                    ).remote(
                        split,
                        self.palim,
                        self.data_id,
                        self.data_eff_id,
                        self.feature_states_id,
                        self.feature_states_index_map_id,
                        self.srcs_id,
                        self.scores_id,
                        self.store_id,
                        None,
                        split_id,
                        self.save_dir,
                        self.score_node,
                        self.only,
                        begin_stage,
                        end_stage,
                        self.nsols,
                        self.kbest,
                        self.mec,
                        self.pruning,
                        self.num_score_workers_per_worker,
                        )
            res_ids.append(return_id)
        return res_ids

    def learn_anytime(
            self,
            data,
            feature_states,
            srcs,
            verbose:bool=False,
            begin_stage:str=None,
            end_stage:str=None,
            scores:dict=None,
            store:dict=None
            ):
        """ Learns the best set of BKFs from the data in an anytime fashion.

        Args:
            :param data: Full database to learn over.
            :type data: np.array
            :param feature_states: List of feature instantiations.
            :type feature_states: list
            :param srcs: A list of source names for each data instance.
            :type srcs: list
        
        Kwargs:
            :param verbose: Whether to print progress bars.
            :type verbose: bool
            :param begin_stage: Stage to begin learning. Passing 'scores' will assume a scores file has been passed and
                will proceed directly to BKF learning.
            :type begin_stage: str
            :param end_stage: Stage to end learning. Passing 'scores' will end learning after all
                scores are calculated.
            :type end_stage: str
            :param scores: Pre-calculated scores dictionary.
            :type scores: dict
            :param store: Pre-calculated store of joint probabilities.
            :type store: dict
        """
        # Update palim if necessary
        if self.palim is None:
            self.palim = len(set([f for f, s in feature_states])) - 1
        self.node_encoding_len = np.log2(len(feature_states))
        report = LearningReport('gobnilp', False)
        report.initialize_bkf_metrics(data.shape[0])
        logger = MPLogger('GobnilpDistributedBackend', logging.INFO, loop_report_time=60)
        # Build feature states index map
        feature_states_index_map = {fs: idx for idx, fs in enumerate(feature_states)}
        # Setup cluster
        pg, bundles = self.setup_cluster()
        logger.info('Setup cluster.')
        # Put data into ray object store
        self.put_data_on_cluster(
                data, 
                feature_states,
                srcs,
                feature_states_index_map,
                scores,
                store,
                )
        logger.info('Put data into object store.')
        logger.info('Setting up work.')
        res_ids = self.setup_anytime_work(data, logger, verbose, pg, bundles, begin_stage, end_stage)
        logger.info('Completed setup.')
        # Run work
        results_finished = 0
        total = len(res_ids)
        logger.info('Collecting Results...')
        logger.initialize_loop_reporting()
        results = []
        while len(res_ids):
            done_ids, res_ids = ray.wait(res_ids)
            res_obj = ray.get(done_ids[0])
            results_finished += 1
            logger.report(i=results_finished, total=total)
            if res_obj is None:
                continue
            if end_stage == 'scores':
                score_objs, _ = res_obj
                results.extend(score_objs)
            else:
                bkb_objs, learn_times = res_obj
                for bkf_obj, learn_time in zip(bkb_objs, learn_times):
                    bkfs, report = self.postprocess_bkf(
                            bkf_obj,
                            learn_time,
                            report,
                            data,
                            feature_states,
                            feature_states_index_map,
                            store,
                            )
                    results.append(bkfs)
        logger.info('Removing placement group.')
        remove_placement_group(pg)
        if end_stage == 'scores':
            if len(results) == 0:
                return []
            return {data_idx: _scores for data_idx, _scores in results}
        else:
            # Sort bkfs into correct order based on data index
            bkfs_to_sort = [(int(bkfs[0].name), bkfs) for bkfs in results]
            bkfs = [bkf for i, bkf in sorted(bkfs_to_sort, key=itemgetter(0))]
            return bkfs, report

    ### Precompute learning strategy functions

    def collect_necessary_joints(self, data, logger, verbose, pg):
        # Only need to calculate joints from unique worlds    
        unique_data = np.unique(data, axis=0)
        # Get max joint sets
        max_joints = set()
        for row in tqdm.tqdm(data, desc='Extracting required max joint sets', disable=not verbose, leave=False):
            max_joints.update(get_max_joint_sets(row, self.palim+1))
        max_joints = np.array(list(max_joints))
        max_joints_id = ray.put(max_joints)
        # Setup up extraction work
        res_ids = []
        for joint_idx in tqdm.tqdm(range(len(max_joints)), desc='Setting up remote calls', disable=not verbose, leave=False):
            res_ids.append(
                    expand_max_joint_set_dist.options(
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            )
                        ).remote(
                            max_joints_id,
                            joint_idx,
                            )
                        )
        # Get extracted joints
        results_finished = 0
        total = len(res_ids)
        logger.info('Collecting Necessary Joints...')
        logger.initialize_loop_reporting()
        necessary_joints = set()
        while(res_ids):
            done_ids, res_ids = ray.wait(res_ids)
            necessary_joints.update(ray.get(done_ids[0]))
            results_finished += 1
            logger.report(i=results_finished, total=total)
        return necessary_joints

    def construct_probability_store(self, necessary_joints, data_len, logger, verbose, pg):
        necessary_joints_id = ray.put(necessary_joints)
        # Setup up calculation work
        res_ids = []
        for joint_idx in tqdm.tqdm(range(len(necessary_joints)), desc='Setting up remote calls', disable=not verbose, leave=False):
            res_ids.append(
                    calc_joint_prob_dist.options(
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            )
                        ).remote(
                            self.data_eff_id,
                            data_len,
                            necessary_joints_id,
                            joint_idx,
                            )
                        )
        # Calculate joints
        results_finished = 0
        total = len(res_ids)
        logger.info('Calculating Necessary Joints...')
        logger.initialize_loop_reporting()
        store = build_probability_store()
        while(res_ids):
            done_ids, res_ids = ray.wait(res_ids)
            prob, joint_idx = ray.get(done_ids[0])
            store[frozenset(necessary_joints[joint_idx])] = prob
            results_finished += 1
            logger.report(i=results_finished, total=total)
        return store

    def construct_master_score_collection(self, necessary_joints):
        score_collection = ScoreCollection(self.node_encoding_len, self.score_node)
        for joint in necessary_joints:
            for x in joint:
                pa_set = set(joint) - {x}
                if len(pa_set) == 0:
                    pa_set = None
                else:
                    pa_set = frozenset(pa_set)
                score_collection.add_score((x, pa_set))
        return score_collection

    def calculate_necessary_scores(self, master_collection, store, logger, verbose, pg):
        store_id = ray.put(store)
        chunk_size = len(master_collection.scores) // self.num_cpus
        # Setup up calculation work
        res_ids = []
        collections = [c for c in master_collection.split_collection(chunk_size)]
        for split_id, split_collection in tqdm.tqdm(
                enumerate(collections),
                desc='Setting up remote calls',
                disable=not verbose,
                leave=False,
                total=len(collections),
                ):
            split_collection.optimize_memory()
            split_collection_id = ray.put(split_collection)
            res_ids.append(
                    calc_score.options(
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            )
                        ).remote(
                            split_collection_id,
                            split_id,
                            self.data_id,
                            self.data_eff_id,
                            self.feature_states_id,
                            self.feature_states_index_map_id,
                            store_id,
                            self.only,
                            )
                        )
        # Calculate scores
        results_finished = 0
        total = len(res_ids)
        logger.info('Calculating Necessary Scores...')
        logger.initialize_loop_reporting()
        scores = [None for _ in range(len(collections))]
        while(res_ids):
            done_ids, res_ids = ray.wait(res_ids)
            _scores, split_id = ray.get(done_ids[0])
            scores[split_id] = _scores
            results_finished += 1
            logger.report(i=results_finished, total=total)
        # Match scores
        all_scores = dict()
        for collection, collection_scores in zip(collections, scores):
            all_scores.update(collection.match_all_scores(collection_scores))
        return all_scores

    def construct_scores(
            self,
            data,
            feature_states,
            feature_states_index_map,
            all_scores,
            store,
            verbose,
            ):
        scores = {}
        total = data.shape[0]
        self.all_scores = all_scores
        self.store = store
        for data_idx in tqdm.tqdm(range(total), desc='Calculating Scores', disable=not verbose, leave=False):
            scores[data_idx] = self.calculate_local_score(
                    data_idx,
                    data,
                    None,
                    feature_states,
                    feature_states_index_map,
                    filepath=None,
                    verbose=verbose,
                    )
        return scores

    def construct_all_scores(self, scores):
        all_scores = dict()
        for _, x_pa_scores in scores.items():
            for x, pa_scores in x_pa_scores.items():
                for pa, score in pa_scores.items():
                    all_scores[(x, pa)] = score
        return all_scores

    def setup_precompute_work(
            self,
            data,
            feature_states,
            feature_states_index_map,
            logger,
            verbose,
            pg,
            bundles,
            begin_stage,
            end_stage,
            store,
            scores,
            ):
        # We need to calculate everything
        if begin_stage is None:
            # Extract necessary joints
            necessary_joints = list(self.collect_necessary_joints(data, logger, verbose, pg))
            # Calculate necessary joint probs
            store = self.construct_probability_store(necessary_joints, data.shape[0], logger, verbose, pg)
            if end_stage == 'store':
                return store
            # Constuct Score Collection based on necessary joints
            score_collection = self.construct_master_score_collection(necessary_joints)
            # Calculate all required scores
            all_scores = self.calculate_necessary_scores(score_collection, store, logger, verbose, pg)
        elif begin_stage == 'store':
            # Extract necessary joints
            necessary_joints = list(self.collect_necessary_joints(data, logger, verbose, pg))
            # Constuct Score Collection based on necessary joints
            score_collection = self.construct_master_score_collection(necessary_joints)
            # Calculate all required scores
            all_scores = self.calculate_necessary_scores(score_collection, store, logger, verbose, pg)
        elif begin_stage == 'scores':
            # Convert to more efficient all scores representation
            all_scores = self.construct_all_scores(scores)
        else:
            raise ValueError(f'Unknown begin stage: {begin_stage}')
        # Construct expected scores if scores is the end stage
        if end_stage == 'scores':
            return self.construct_scores(data, feature_states, feature_states_index_map, all_scores, store, verbose), store
        # Put all_scores in the object store
        all_scores_id = ray.put(all_scores)
        # Now construct bkf learning work
        row_splits = self.split_data_over_cluster(data, logger)
        # Split row indices over number of available CPUs
        res_ids = []
        split_counter = 0
        bundle_idx = 0
        for split_id, split in tqdm.tqdm(enumerate(row_splits), desc='Setting up remote calls', disable=not verbose, leave=False, total=len(row_splits)):
            split_counter += 1
            if split_counter > self.num_workers_per_node:
                # Move on to the next bundle
                split_counter = 0
                bundle_idx += 1
            # Build BKF
            return_id = learn_distributed.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=bundle_idx,
                        )
                    ).remote(
                        split,
                        self.palim,
                        self.data_id,
                        self.data_eff_id,
                        self.feature_states_id,
                        self.feature_states_index_map_id,
                        self.srcs_id,
                        self.scores_id,
                        self.store_id,
                        all_scores_id,
                        split_id,
                        self.save_dir,
                        self.score_node,
                        self.only,
                        None,
                        None,
                        self.nsols,
                        self.kbest,
                        self.mec,
                        self.pruning,
                        self.num_score_workers_per_worker,
                        )
            res_ids.append(return_id)
        return res_ids

    def learn_precompute(
            self,
            data,
            feature_states,
            srcs,
            verbose:bool=False,
            begin_stage:str=None,
            end_stage:str=None,
            scores:dict=None,
            store:dict=None,
            ):
        """ Learns the best set of BKFs from the data by precomputing all intermediate
        data such as all necessary joint probs, then scorse, and then learning all BKFs.

        Args:
            :param data: Full database to learn over.
            :type data: np.array
            :param feature_states: List of feature instantiations.
            :type feature_states: list
            :param srcs: A list of source names for each data instance.
            :type srcs: list
        
        Kwargs:
            :param verbose: Whether to print progress bars.
            :type verbose: bool
            :param begin_stage: Stage to begin learning. Passing 'scores' will assume a scores obje has been passed and
                will proceed directly to BKF learning.
            :type begin_stage: str
            :param end_stage: Stage to end learning. Passing 'scores' will end learning after all
                scores are calculated.
            :type end_stage: str
            :param scores: Pre-calculated scores dictionary.
            :type scores: dict
            :param store: Pre-calculated store of joint probabilities.
            :type store: dict
            :param scores_filepath: Path where to save scores.
            :type scores_filepath: str
        """
        # Update palim if necessary
        if self.palim is None:
            self.palim = len(set([f for f, s in feature_states])) - 1
        self.node_encoding_len = np.log2(len(feature_states))
        report = LearningReport('gobnilp', False)
        report.initialize_bkf_metrics(data.shape[0])
        logger = MPLogger('GobnilpDistributedBackend', logging.INFO, loop_report_time=60)
        # Build feature states index map
        feature_states_index_map = {fs: idx for idx, fs in enumerate(feature_states)}
        # Setup cluster
        pg, bundles = self.setup_cluster()
        logger.info('Setup cluster.')
        # Put data into ray object store
        self.put_data_on_cluster(
                data, 
                feature_states,
                srcs,
                feature_states_index_map,
                None,
                store,
                )
        logger.info('Put data into object store.')
        logger.info('Setting up work.')
        if end_stage == 'store' or end_stage == 'scores':
            return self.setup_precompute_work(
                    data,
                    feature_states,
                    feature_states_index_map,
                    logger, 
                    verbose,
                    pg,
                    bundles,
                    begin_stage,
                    end_stage, 
                    store,
                    scores,
                    )
        # Else we will learn BKFs
        res_ids = self.setup_precompute_work(
                data,
                feature_states,
                feature_states_index_map,
                logger, 
                verbose,
                pg,
                bundles,
                begin_stage,
                end_stage, 
                store,
                scores,
                )
        logger.info('Completed setup.')
        # Run work
        results_finished = 0
        total = len(res_ids)
        logger.info('Collecting Results...')
        logger.initialize_loop_reporting()
        results = []
        while len(res_ids):
            done_ids, res_ids = ray.wait(res_ids)
            res_obj = ray.get(done_ids[0])
            results_finished += 1
            logger.report(i=results_finished, total=total)
            if res_obj is None:
                continue
            else:
                bkb_objs, learn_times = res_obj
                for bkf_obj, learn_time in zip(bkb_objs, learn_times):
                    bkfs, report = self.postprocess_bkf(
                            bkf_obj,
                            learn_time,
                            report,
                            data,
                            feature_states,
                            feature_states_index_map,
                            store,
                            )
                    results.append(bkfs)
        logger.info('Removing placement group.')
        remove_placement_group(pg)
        # Sort bkfs into correct order based on data index
        bkfs_to_sort = [(int(bkfs[0].name), bkfs) for bkfs in results]
        bkfs = [bkf for i, bkf in sorted(bkfs_to_sort, key=itemgetter(0))]
        return bkfs, report

    def postprocess_bkf(
            self,
            bkf_objs,
            learn_time,
            report,
            data,
            feature_states,
            feature_states_index_map,
            store,
            ):
        # Load bkf object
        bkfs = [BKB.loads(bkf_obj) for bkf_obj in bkf_objs]
        data_idx = int(bkfs[0].name)
        # Calculate metrics
        report.add_bkf_metrics(data_idx, learn_time=learn_time)
        # Get scores for this bkf to put in report
        _dscore, _mscore = bkfs[0].score(
                data,
                feature_states,
                self.score,
                feature_states_index_map=feature_states_index_map, 
                only='both',
                store=store,
                )
        report.add_bkf_metrics(
                data_idx,
                model_score=_mscore,
                data_score=_dscore,
                )
        return bkfs, report


#### Distrbuted Remote Functions

@ray.remote
def expand_max_joint_set_dist(max_joints, joint_idx):
    """ Simple distributed wrapper of pybkb.util.probability.expand_max_joint_set().
    
    Args:
        :param max_joints: The list of max joints to expand.
        :type max_joints: list
        :param joint_idx: The specific index in the max_joints list to expand.
        :type joint_idx: int
    """

    return expand_max_joint_set(max_joints[joint_idx])

@ray.remote
def calc_joint_prob_dist(data_eff_id, data_len, necessary_joints_id, joint_idx):
    prob, _ = joint_prob_eff(nb.typed.List(data_eff_id), data_len, parent_state_indices=list(necessary_joints_id[joint_idx]))
    return (prob, joint_idx)

@ray.remote
def calc_score(
            split_collection_id,
            split_id,
            data_id,
            data_eff_id,
            feature_states_id,
            feature_states_index_map_id,
            store_id,
            only,
            ):
    logger = MPLogger('Score Worker', logging.INFO, id=split_id, loop_report_time=60)
    return (
            split_collection_id.calculate_scores_vector(
                data_id,
                nb.typed.List(data_eff_id),
                feature_states_id,
                feature_states_index_map_id,
                store_id,
                only,
                logger,
                ),
            split_id
            )

@ray.remote(num_cpus=1)
def learn_distributed(
        row_indices,
        palim,
        data_id,
        data_eff_id,
        feature_states_id,
        feature_states_index_map_id,
        srcs_id,
        scores_id,
        store_id,
        all_scores_id,
        split_id,
        save_dir,
        score_node,
        only,
        begin_stage,
        end_stage,
        nsols,
        kbest,
        mec,
        pruning,
        num_workers,
        ):
    """ Function that scores and/or learns bkfs distrbutively over a cluster.

        Args:
            :param row_indices: The row indices of the dataset that this worker will learn/score.
            :type row_indices: list
            :param palim: Limit on the number of parent sets. 
            :type palim: int
            :param data_id: The ray object reference pointing to the full database to learn over.
            :type data_id: ray.ObjectRef
            :param data_eff_id: The ray object reference pointing to the bkb data formatted for efficiency as given by
                pybkb.learn.backends.bkb.gobnilp.BKBGobnilpBackend.format_data_for_probs function.
            :type data_eff_id: ray.ObjectRef
            :param feature_states_id: The ray object reference pointing to the list of feature instantiations.
            :type feature_states_id: ray.ObjectRef
            :param feature_states_index_map_id: The ray object reference poiting to the dictionary mapping
                feature state tuples to appropriate column index in the data matrix.
            :type feature_states_index_map_id: ray.ObjectRef
            :param srcs_id: The ray object reference pointing to the list of source names for each data instance.
            :type srcs_id: ray.ObjectRef
            :param scores_id: The ray object reference pointing to the pre-calculated scores dictionary.
            :type scores: ray.ObjectRef
            :param scores_id: The ray object reference pointing to the pre-calculated all scores dictionary, a simpler hash table.
            :type scores: ray.ObjectRef
            :param store: The ray object reference pointing to the pre-calculated store of joint probabilities.
            :type store: ray.ObjectRef
            :param split_id: The data split id to be used to identify the worker.
            :type split_id: int
            :param save_dir: The directory to either save scores, bkfs or both.
            :type save_dir: str
            :param score_node: Node scoring object to be used to score.
            :type score_node: pybkb.scores.ScoreNode
            :param only: Return only the data score or model score or both. Options: data, model, None. Defaults to None which means both.
            :type only: str
            :param begin_stage: Stage to begin learning. Passing 'scores' will assume a scores file has been passed and
                will proceed directly to BKF learning.
            :type begin_stage: str
            :param end_stage: Stage to end learning. Passing 'scores' will end learning after all
                scores are calculated.
            :type end_stage: str
            :param nsols: Number of BKBs to learn per example.
            :type nsols: int
            :param kbest: Whether the nsols learned BKBs should be a highest scoring set of nsols BKBs.
            :type kbest: bool
            :param mec: Whether only one BKB per Markov equivalence class should be feasible.
            :type mec: bool
            :param pruning: Whether not to include parent sets which cannot be optimal when acyclicity is the only constraint.
            :type pruning: bool
    """
    objs, run_times = [], []
    logger = MPLogger('Learning Worker', logging.INFO, id=split_id, loop_report_time=60)
    logger.info(f'Starting to work on {len(row_indices)} examples.')
    logger.initialize_loop_reporting()
    store = build_probability_store()
    if all_scores_id:
        all_scores_dict = all_scores_id
    else:
        all_scores_dict = {}
    if num_workers:
        pool = Pool(num_workers)
    else:
        pool = None
    for i, row_idx in enumerate(row_indices):
        # Calculate scores and store
        start_time = time.time()
        logger.debug('Scoring example.')
        if begin_stage == 'scores':
            logger.debug('Using pre-calculated scores.')
            scores = scores_id[row_idx]
        else:
            scores, store, all_scores_dict = BKBGobnilpBackend.calculate_local_score_static(
                    row_idx,
                    data_id,
                    nb.typed.List(data_eff_id),
                    feature_states_id,
                    palim,
                    score_node,
                    feature_states_index_map_id,
                    store=store,
                    only=only,
                    all_scores=all_scores_dict,
                    logger=logger,
                    pool=pool,
                    )
        score_time = time.time() - start_time
        logger.debug(f'Calculated or found scores in {score_time} seconds.')
        if save_dir:
            # Save out the scores csv file.
            BKBGobnilpBackend.write_scores_to_file(
                    scores,
                    row_idx,
                    os.path.join(save_dir, f'scores-{row_idx}-{srcs_id[row_idx]}.csv'),
                    open_mode='w',
                    )
            if end_stage == 'scores':
                continue
        # If the end stage is scores, then don't do learning
        if end_stage == 'scores':
            objs.append((row_idx, scores))
            run_times.append(score_time)
            continue
        logger.debug('Starting to learn on example.')
        m, learn_time = BKBGobnilpBackend.learn_dag_from_scores(
                scores,
                data_id,
                row_idx,
                nsols,
                kbest,
                pruning,
                mec,
                )
        if save_dir:
            # Write out the model used to learn the corresponding BKF.
            m.write(os.path.join(save_dir, f'model-{row_idx}-{srcs_id[row_idx]}.mps'))
        # Convert learned DAG to BKF (learned_bn of gobnilp is a subclass of networkx.DiGraph)
        try:
            bkfs = BKBGobnilpBackend.convert_dags_to_bkf(
                    m.learned_bns,
                    m.learned_scores,
                    str(row_idx),
                    data_id,
                    feature_states_id,
                    feature_states_index_map_id,
                    store=store,
                    )
        except:
            print(f'Problem on data index: {row_idx}.')
            continue
        logger.report(i=i, total=len(row_indices))
        if save_dir:
            for i, bkf in enumerate(bkfs):
                src = srcs_id[row_idx]
                bkf.save(os.path.join(save_dir, f'learned-bkf-{row_idx}-{src}-{i}.bkb'))
            continue
        bkfs_dumps = []
        for i, bkf in enumerate(bkfs):
            bkfs_dumps.append(bkf.dumps())
        # Transmit BKF efficiently using inherent numpy representation and reload on other end
        objs.append(bkfs_dumps)
        run_times.append(learn_time)
    # Close worker pool if there was one.
    if pool:
        pool.close()
        pool.join()
    logger.info('Finished learning.')
    if save_dir:
        logger.info('Since a save directory was passed, objects will not be passed to main process.')
        return None
    return (objs, run_times)
