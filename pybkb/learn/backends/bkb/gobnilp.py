import os, csv
import itertools
import contextlib
import time
import tqdm
import numpy as np
import numba as nb
from operator import itemgetter
from pygobnilp.gobnilp import Gobnilp
from gurobipy import GRB, read
from multiprocessing import Pool
from scipy.special import comb
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from pybkb.utils.probability import *
from pybkb.scores import *
from pybkb.learn.report import LearningReport
from pybkb.bkb import BKB


class BKBGobnilpBackend:
    def __init__(
            self,
            score:str,
            palim:int=None,
            only:str=None,
            ) -> None:
        """ BKB Gobnilp DAG learning backend.

        Args:
            :param score: Name of scoring function. Choices: mdl_mi, mdl_ent.
            :type score: str

        Kwargs:
            :param palim: Limit on the number of parent sets. 
            :type palim: int
            :param only: Return only the data score or model score or both. Options: data, model, None. Defaults to None which means both.
            :type only: str
        """
        if score == 'mdl_mi':
            self.score_node = MdlMutInfoScoreNode
        elif score == 'mdl_ent':
            self.score_node = MdlEntScoreNode
        else:
            raise ValueError(f'Unknown score: {score}')
        self.palim = palim
        self.score = score
        self.only = only 
        self.store = build_probability_store()
        self.all_scores = {}
    
    def learn(
            self,
            data:np.array,
            feature_states:list,
            srcs:list,
            verbose:bool=False,
            scores:dict=None,
            store:dict=None,
            begin_stage:str=None,
            end_stage:str=None,
            nsols:int=1,
            kbest:bool=True,
            mec:bool=False,
            pruning:bool=False,
            num_workers:int=None,
            scores_filepath:str=None,
            ):
        """ Learns the best set of BKFs from the data.

        Args:
            :param data: Full database to learn over.
            :type data: np.array
            :param feature_states: List of feature instantiations.
            :type feature_states: list
            :param srcs: A list of source names for each data row.
            :type srcs: list

        Kwargs:
            :param verbose: Whether to print progress bars.
            :type verbose: bool
            :param scores: Pre-calculated scores dictionary.
            :type scores: dict
            :param store: Pre-calculated store of joint probabilities.
            :type store: dict
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
            :param num_workers: Number of workers in a distributed pool to calculate scores and joint probabilities.
            :type num_workers: int
            :param scores_filepath: Optional filepath where local scores will be written. Defaults None.
            :type scores_filepath: str
        """
        # Initialize report
        report = LearningReport('gobnilp', False)
        report.initialize_bkf_metrics(data.shape[0])
        # Build feature states index map
        feature_states_index_map = {fs: idx for idx, fs in enumerate(feature_states)}
        self.feature_states_index_map = feature_states_index_map
        # Build more effective data structures
        data_eff = self.format_data_for_probs(data, verbose=verbose)
        self.data_eff = data_eff
        # Reset store unless one is passed
        if store is None:
            self.store = build_probability_store()
        else:
            self.store = store
        # Update palim if necessary
        if self.palim is None:
            self.palim = len(set([f for f, s in feature_states])) - 1
        # Create score_pool
        if num_workers is not None:
            self.score_pool = Pool(num_workers)
        else:
            self.score_pool = None
        # Determine end stage
        if end_stage == 'scores':
            scores = self.calculate_all_local_scores(
                    data,
                    feature_states,
                    data_eff,
                    filepath=scores_filepath,
                    verbose=verbose,
                    )
            return scores, self.all_scores, self.store
        elif end_stage is None:
            pass
        else:
            raise ValueError(f'Unknown end stage option: {end_stage}.')
        # Build optimal bkfs
        self.scores = []
        bkfs = []
        self.models = []
        for data_idx in tqdm.tqdm(range(data.shape[0]), desc='Learning Fragments', disable=not verbose):
            # Calculate local scores
            if begin_stage == 'scores':
                example_scores = scores[data_idx]
            else:
                example_scores = self.calculate_local_score(data_idx, data, data_eff, feature_states, feature_states_index_map)
            self.scores.append(example_scores)
            # Update report
            report.update_from_bkf_store(data_idx, self.store)
            # Learn best DAG from local scores using Gobnilp
            m, report = self.learn_dag_from_scores(
                    example_scores,
                    data,
                    data_idx,
                    nsols,
                    kbest,
                    pruning,
                    mec,
                    report,
                    )
            # Store gurobi models for regression tests.
            self.models.append(m)
            '''ToRemove
            # Learn the best DAG from these local scores using Gobnilp
            # Redirect output so we don't have to see this
            f = open(os.devnull, 'w')
            with contextlib.redirect_stdout(f):
                m = Gobnilp()
                # Start the learning but stop before learning to add constraints
                m.learn(local_scores_source=example_scores, end='MIP model', nsols=nsols, kbest=kbest, pruning=pruning, mec=mec)
                # Grab all the adjacency variables
                adj = [v for p, v in m.adjacency.items()]
                # Add a constraint that at the DAG must be connected (need to subtract one to make a least a tree)
                m.addLConstr(sum(adj), GRB.GREATER_EQUAL, np.sum(data[data_idx,:]) - 1)
            # Close devnull file as to not get resource warning
            f.close()
            # Learn the DAG
            report.start_timer()
            m.learn(local_scores_source=example_scores, start='MIP model')
            # Add learning time to report
            report.add_bkf_metrics(data_idx, learn_time=report.end_timer())
            '''
            # Convert learned DAG to BKF (learned_bn of gobnilp is a subclass of networkx.DiGraph)
            bkfs.append(
                    self.convert_dags_to_bkf(
                        m.learned_bns,
                        m.learned_scores,
                        str(data_idx),
                        data,
                        feature_states,
                        feature_states_index_map,
                        self.store
                        )
                    )
            # Get scores for this bkf to put in report, they'll all have the same score
            _dscore, _mscore = bkfs[-1][0].score(
                    data,
                    feature_states,
                    self.score,
                    feature_states_index_map=feature_states_index_map, 
                    only='both',
                    store=self.store,
                    )
            report.add_bkf_metrics(
                    data_idx,
                    model_score=_mscore,
                    data_score=_dscore,
                    bns=m.learned_bns,
                    )
        # Close down the pool, if used.
        if num_workers is not None:
            self.score_pool.close()
            self.score_pool.join()
        return bkfs, report

    @staticmethod
    def learn_dag_from_scores(
            example_scores,
            data,
            data_idx,
            nsols,
            kbest,
            pruning,
            mec,
            report=None,
            ):
        """ Function to learn the best example DAG from scores using GOBNILP.

        Args:
            :param example_scores: Parent set scores for the given data instance (example).
            :type example_scores: dict
            :param data: Full database to learn over.
            :type data: np.array
            :param data_idx: The row index of the example that is being learned.
            :type data_idx: int
            :param nsols: Number of BKBs to learn per example.
            :type nsols: int
            :param kbest: Whether the nsols learned BKBs should be a highest scoring set of nsols BKBs.
            :type kbest: bool
            :param mec: Whether only one BKB per Markov equivalence class should be feasible.
            :type mec: bool
            :param pruning: Whether not to include parent sets which cannot be optimal when acyclicity is the only constraint.
            :type pruning: bool

        Kwargs:
            :param report: The learning report to update.
            :type report: pybkb.learn.report.LearningReport
        """
        # Learn the best DAG from these local scores using Gobnilp
        # Redirect output so we don't have to see this
        f = open(os.devnull, 'w')
        with contextlib.redirect_stdout(f):
            m = Gobnilp()
            # Start the learning but stop before learning to add constraints
            m.learn(local_scores_source=example_scores, end='MIP model', nsols=nsols, kbest=kbest, pruning=pruning, mec=mec)
            # Grab all the adjacency variables
            adj = [v for p, v in m.adjacency.items()]
            # Add a constraint that at the DAG must be connected (need to subtract one to make a least a tree)
            m.addLConstr(sum(adj), GRB.GREATER_EQUAL, np.sum(data[data_idx,:]) - 1)
        # Close devnull file as to not get resource warning
        f.close()
        # Learn the DAG
        start_time = time.time()
        m.learn(local_scores_source=example_scores, start='MIP model', gurobi_output=False, verbose=0)
        # Add learning time to report
        learn_time = time.time() - start_time
        if report:
            report.add_bkf_metrics(data_idx, learn_time=learn_time)
            return m, report
        return m, learn_time

    def calculate_all_local_scores(
            self,
            data:np.array,
            feature_states:list,
            data_eff:list,
            filepath:str=None,
            verbose:bool=False,
            logger=None,
            reset:bool=True,
            ) -> dict:
        """ Generates local scores for Gobnilp optimization
        
        Args:
            :param data: Full database to learn over.
            :type data: np.array
            :param feature_states: List of feature instantiations.
            :type feature_states: list
            :param data_eff: The bkb data formatted for efficiency as given by
                pybkb.learn.backends.bkb.gobnilp.BKBGobnilpBackend.format_data_for_probs function.
            :type data_eff: list

        Kwargs:
            :param filepath: Optional filepath where local scores will be written. Defaults None.
            :type filepath: str
            :param verbose: Whether to print progress bars.
            :type verbose: bool
            :param logger: A MPLogger used for distributed mode logging.
            :type logger: pybkb.utils.mp.MPLogger
            :param reset: Whether to reset the store and scores hashtables.
            :type reset: bool
        """
        # Build feature states index map
        feature_states_index_map = {fs: idx for idx, fs in enumerate(feature_states)}
        if reset:
            # Reset store
            self.store = build_probability_store()
            # Calculate scores
            self.all_scores = {}
        if filepath:
            # Initialize file 
            with open(filepath, 'w') as f_:
                writer = csv.writer(f_)
                writer.writerow(['Data_Index', 'Head_Feature_State', 'Score', 'Parent_Feature_States']) 
        scores = {}
        total = data.shape[0]
        for data_idx in tqdm.tqdm(range(total), desc='Calculating Scores', disable=not verbose, leave=False):
            scores[data_idx] = self.calculate_local_score(
                    data_idx,
                    data,
                    data_eff,
                    feature_states,
                    feature_states_index_map,
                    filepath=filepath,
                    all_scores=self.all_scores,
                    verbose=verbose,
                    )
            if logger is not None:
                logger.report(i=data_idx+1, total=total)
        return scores
    
    def calculate_local_score(
            self,
            data_idx:int,
            data:np.array,
            data_eff,
            feature_states:list,
            feature_states_index_map:dict,
            filepath:str=None,
            verbose=False,
            ) -> dict:
        """ Generates a data instance's local score for Gobnilp optimization, a class wrapper for 
            BKBGobnilpBackend.calculate_local_score_static.
        
        Args:
            :param data_idx: Row index of data instance to calculate local scores.
            :type data_idx: int
            :param data: Full database to learn over.
            :type data: np.array
            :param data_eff: The bkb data formatted for efficiency as given by
                pybkb.learn.backends.bkb.gobnilp.BKBGobnilpBackend.format_data_for_probs function.
            :type data_eff: list
            :param feature_states: List of feature instantiations.
            :type feature_states: list
            :param feature_states_index_map: A dictionary mapping feature state tuples to appropriate column index in the data matrix.
            :type feature_states_index_map: dict

        Kwargs:
            :param filepath: Optional filepath where local scores will be written. Defaults None.
            :type filepath: str
            :param all_scores: A dictionary holding all previously calculated scores.
            :param all_scores: dict
            :param verbose: Whether to print progress bars.
            :type verbose: bool
        """
        score, self.store, self.all_scores = self.calculate_local_score_static(
                data_idx,
                data,
                data_eff,
                feature_states,
                self.palim,
                self.score_node,
                feature_states_index_map,
                filepath,
                self.store,
                'data',#self.only,
                self.all_scores,
                verbose,
                None,
                self.score_pool,
                )
        return score
                
    @staticmethod
    def calculate_local_score_static(
            data_idx,
            data,
            data_eff,
            feature_states:list,
            palim,
            score_node,
            feature_states_index_map:dict,
            filepath:str=None,
            store:dict=None,
            only:str=None,
            all_scores:dict=None,
            verbose:bool=False,
            logger=None,
            pool=None,
            ):
        """ Generates a data instance's local score for Gobnilp optimization. Static method to be used externally.
        
        Args:
            :param data_idx: Row index of data instance to calculate local scores.
            :type data_idx: int
            :param data: Full database to learn over.
            :type data: np.array
            :param data_eff: The dataset formatted using BKBGobnilpBackend.format_data_for_probs.
            :type data_eff: dict
            :param feature_states: List of feature instantiations.
            :type feature_states: list
            :param palim: Parent set limit. None means that there is no limit.
            :type palim: int
            :param score_node: The score node object.
            :type score_node: pybkb.learn.scores.ScoreNode
            :param feature_states_index_map: A dictionary mapping feature state tuples to appropriate column index in the data matrix.
            :type feature_states_index_map: dict

        Kwargs:
            :param filepath: Optional filepath where local scores will be written. Defaults None.
            :type filepath: str
            :param store: The probability store for the various joint probability calculations.
            :type store: dict
            :param only: Return only the data score or model score or both. Options: data, model, None. Defaults to None which means both.
            :type only: str
            :param all_scores: A dictionary holding all previously calculated scores.
            :param all_scores: dict
            :param verbose: Whether to print progress bars.
            :type verbose: bool
            :param logger: A MPLogger used for distributed mode logging.
            :type logger: pybkb.utils.mp.MPLogger
            :param pool: The multiprocessing pool to use to distribute joint probability and score 
                calculations.
            :type pool: mp.Pool

        """
        # Setup all scores if not passed
        if all_scores is None:
            all_scores = {}
        # Collect feature instantiations in this data instance
        fs_indices = np.argwhere(data[data_idx,:] == 1).flatten()
        # Initialize scores
        scores = defaultdict(dict)
        # Initialize store if not passed
        if store is None:
            store = build_probability_store()
        # Calculate node encoding length
        node_encoding_len = np.log2(len(feature_states))
        # Collect score nodes
        score_collection = BKBGobnilpBackend.collect_score_nodes(
            palim,
            score_node,
            fs_indices,
            node_encoding_len,
            verbose,
            None,
            )
        if logger is not None:
            logger.debug('Calculating joint probs')
        store = BKBGobnilpBackend.calculate_required_joint_probs(
                score_collection,
                data_eff,
                store,
                data.shape[0],
                verbose,
                pool,
                )
        # Calculate scores
        if logger is not None:
            logger.debug('Calculating scores probs')
        scores, store, all_scores = score_collection.calculate_scores(
                data, 
                data_eff, 
                feature_states, 
                store, 
                feature_states_index_map, 
                only, 
                verbose, 
                all_scores
                )
        if filepath:
            BKBGobnilpBackend.write_scores_to_file(scores, data_idx, filepath, write_header=False)
        return scores, store, all_scores

    @staticmethod
    def format_data_for_probs(
            data:np.array,
            verbose:bool=False,
            pickleable:bool=False,
            ):
        """ Formats standard BKB data into an efficient data structure for calculating joint
            probabilities.

        Args:
            :param data: A standard BKB dataset. Should be a one-hot encoded NxM matrix such that N is 
                the number of examples and M is the number of feature states.
            :type data: np.array
        
        Kwargs:
            :param verbose: Whether to print progress bars.
            :type verbose: bool
            :param pickleable: Whether to return normal python list (pickleable) or numba list (not currently pickleable).
                If using python list it must be cast back to numba list before calling joint probability calculator.
            :type pickleable: bool
        """
        row_indices, col_indices = np.nonzero(data)
        _data_eff = [set() for _ in range(data.shape[1])]
        # To be compiled with Numba need a list of sets representation
        for row_idx, fs_idx in tqdm.tqdm(
                zip(row_indices, col_indices), 
                desc='Formatting data', 
                disable=not verbose, 
                leave=False,
                total=len(row_indices),
                ):
            _data_eff[fs_idx].add(row_idx)
        # Recast each set as a numpy array
        data_eff = []
        for row_eff in _data_eff:
            data_eff.append(np.array(list(row_eff), dtype=np.float64))
        if pickleable:
            return data_eff
        return nb.typed.List(data_eff)

    @staticmethod
    def collect_rvi_score_nodes(
            rvis,
            palim,
            score_node_obj,
            all_rvis,
            node_encoding_len,
            verbose=False,
            position=0
            ):
        """ Function to collect a score node collection given a set of Random Variable Instantiations (RVIs).
        
        Args:
            :param rvis: Random Variable Instantiations (targets) of the score nodes to collect.
            :type rvis: list, set, id
            :param palim: Limit on the number of parent sets. 
            :type palim: int
            :param score_node_obj: The base score node object to use for scoring parent sets.
            :type score_node_obj: pybkb.scores.ScoreNode
            :param all_rvis: All RVIs that can be possible parents.
            :type all_rvis: list, set
            :param node_encoding_len: The encoding length of a single score node.
            :type node_encoding_len: float

        Kwargs:
            :param verbose: Whether to print progress bars.
            :type verbose: bool
            :param position: The position of the tqdm progress bar.
            :type position: int
        """
        score_collection = ScoreCollection(node_encoding_len, score_node_obj)
        if type(rvis) not in [list, set]:
            rvis = [rvis]
        for rvi in tqdm.tqdm(rvis, disable=not verbose, desc='Collecting scores', leave=False, position=position):
            # Need to add one to palim due to python zero indexing
            for i in range(palim + 1):
                # No parent set score
                if i == 0:
                    pa_set_iter = [[]]
                else:
                    pa_set_iter = itertools.combinations(set.difference(all_rvis, {rvi}), r=i)
                #total_combos = comb(len(all_rvis) - 1, i)
                for pa_set in pa_set_iter:
                    score_collection.add_score((rvi, frozenset(pa_set)))
        return score_collection.scores

    @staticmethod
    def collect_score_nodes(
            palim,
            score_node_obj,
            all_rvis,
            node_encoding_len,
            verbose,
            pool,
            ):
        """ Function to collect a score node collection given for ALL Random Variable Instantiations (RVIs).
        
        Args:
            :param palim: Limit on the number of parent sets. 
            :type palim: int
            :param score_node_obj: The base score node object to use for scoring parent sets.
            :type score_node_obj: pybkb.scores.ScoreNode
            :param all_rvis: All RVIs that can be possible parents.
            :type all_rvis: list, set
            :param node_encoding_len: The encoding length of a single score node.
            :type node_encoding_len: float
            :param verbose: Whether to print progress bars.
            :type verbose: bool
            :param pool: The multiprocessing pool to use to distribute joint probability and score 
                calculations.
            :type pool: mp.Pool
        """
        if pool is not None:
            return BKBGobnilpBackend.collect_score_nodes_distributed(
                palim,
                score_node_obj,
                all_rvis,
                node_encoding_len,
                verbose,
                pool
                )
        # Collect Score Nodes (Not distributed)
        score_collection = ScoreCollection(node_encoding_len, score_node_obj)
        all_rvis = set(all_rvis)
        for rvi in tqdm.tqdm(all_rvis, desc='Collecting Score Nodes', leave=False, disable=not verbose):
            scores = BKBGobnilpBackend.collect_rvi_score_nodes(
                    rvi,
                    palim,
                    score_node_obj,
                    all_rvis,
                    node_encoding_len,
                    )
            score_collection.merge_scores(scores)
            del scores
        return score_collection

    @staticmethod
    def collect_score_nodes_distributed(
            palim,
            score_node_obj,
            all_rvis,
            node_encoding_len,
            verbose,
            pool,
            ):
        """ Function to distrbutively collect a score node collection given for ALL Random Variable Instantiations (RVIs).
        
        Args:
            :param palim: Limit on the number of parent sets. 
            :type palim: int
            :param score_node_obj: The base score node object to use for scoring parent sets.
            :type score_node_obj: pybkb.scores.ScoreNode
            :param all_rvis: All RVIs that can be possible parents.
            :type all_rvis: list, set
            :param node_encoding_len: The encoding length of a single score node.
            :type node_encoding_len: float
            :param verbose: Whether to print progress bars.
            :type verbose: bool
            :param pool: The multiprocessing pool to use to distribute joint probability and score 
                calculations.
            :type pool: mp.Pool
        """
        # Initialize
        score_collection = ScoreCollection(node_encoding_len, score_node_obj)
        # Callback fn
        def mp_callback(res):
            score_collection.merge_scores(res)
        # Setup async results
        num_workers = pool._processes
        splits = np.array_split(all_rvis, num_workers)
        results = []
        for i, rvi_split in enumerate(splits):
            results.append(
                    pool.apply_async(
                        BKBGobnilpBackend.collect_rvi_score_nodes,
                        (list(rvi_split), palim, score_node_obj, set(all_rvis), node_encoding_len, verbose, i+1),
                        callback=mp_callback,
                        )
                    )
        # Collect 
        with tqdm.tqdm(desc='Collecting Score Nodes', total=len(splits), leave=False, disable=not verbose) as pbar:
            for r in results:
                r.wait()
                pbar.update(1)
        return score_collection

    @staticmethod
    def calculate_required_joint_probs(
            score_collection,
            data_eff,
            store,
            data_len,
            verbose,
            pool
            ):
        """ Function to calculate all necessary joint probabilities required by a score collection.
        
        Args:
            :param score_collection: A collection of scores to be calculated.
            :type score_collection: pybkb.scores.ScoreCollection
            :param data_eff: The dataset formatted using BKBGobnilpBackend.format_data_for_probs.
            :type data_eff: dict
            :param store: Probability store.
            :type store: dict
            :param data_len: Length of the dataset, i.e. number of examples.
            :type data_len: int
            :param verbose: Whether to print progress bars.
            :type verbose: bool
            :param pool: The multiprocessing pool to use to distribute joint probability and score 
                calculations.
            :type pool: mp.Pool
        """
        if pool is not None:
            return BKBGobnilpBackend.calculate_required_joint_probs_distributed(
                    score_collection,
                    data_eff,
                    store,
                    data_len,
                    verbose,
                    pool,
                    )
        for indices in score_collection.extract_joint_probs(verbose):
            _, store = joint_prob_eff(data_eff, data_len, parent_state_indices=indices, store=store)
        return store

    @staticmethod
    def calculate_required_joint_probs_distributed(
            score_collection,
            data_eff,
            store, 
            data_len, 
            verbose, 
            pool, 
            chunk_size=1e6
            ):
        """ Function to calculate all necessary joint probabilities required by a score collection.
        
        Args:
            :param score_collection: A collection of scores to be calculated.
            :type score_collection: pybkb.scores.ScoreCollection
            :param data_eff: The dataset formatted using BKBGobnilpBackend.format_data_for_probs.
            :type data_eff: dict
            :param store: Probability store.
            :type store: dict
            :param data_len: Length of the dataset, i.e. number of examples.
            :type data_len: int
            :param verbose: Whether to print progress bars.
            :type verbose: bool
            :param pool: The multiprocessing pool to use to distribute joint probability and score 
                calculations.
            :type pool: mp.Pool

        Kwargs:
            :param chunk_size: Size of joint probabilities that should be sent to each worker.
            :type chunk_size: int
        """
        # Callback fn
        def mp_callback(_store):
            ncalls = _store.pop('__ncalls__')
            nlookups = _store.pop('__nhashlookups__')
            store.update(_store)
            del _store
            store['__ncalls__'] += ncalls
            store['__nhashlookups__'] += nlookups
        # Setup async results
        results = []
        num_workers = pool._processes
        max_worker_chunk_size = len(score_collection.scores) // num_workers
        if chunk_size is None or chunk_size > max_worker_chunk_size:
            chunk_size = max_worker_chunk_size
        j = 1
        for sc in score_collection.split_collection(chunk_size):
            results.append(
                    pool.apply_async(
                        joint_probs_eff_from_sc,
                        (data_eff, data_len, sc, verbose, j, None),
                        callback=mp_callback,
                        )
                    )
            j += 1
            if j == num_workers + 1:
                j = 1
        # Collect 
        with tqdm.tqdm(desc='Calculating Required Joints', total=len(results), leave=False, disable=not verbose) as pbar:
            for r in results:
                r.wait()
                pbar.update(1)
        return store

    @staticmethod
    def write_scores_to_file(scores, data_idx, filepath, open_mode='a', write_header=True):
        """ Function to write scores to file in csv format such that column 1 is the data row index of the
        source, second row is score value, third row is head feature state index, and the remaining rows are
        the parent feature state indices.

        Args:
            :param scores: Local scores of an example.
            :type scores: dict
            :param data_idx: The row index in the dataset corresponding to the example.
            :type data_idx: int
            :param open_mode: The mode to open the file, i.e. 'a' or 'w', etc.
            :type open_mode: str
            :param write_header: Whether to write the standard scores csv file header.
            :type write_header: bool
        """
        with open(filepath, open_mode) as f_:
            writer = csv.writer(f_)
            if write_header:
                writer.writerow(['Data_Index', 'Head_Feature_State', 'Score', 'Parent_Feature_States']) 
            for x, paset_scores in scores.items():
                for pa_set, score in paset_scores.items():
                    writer.writerow(
                            [data_idx, x, score] + list(pa_set)
                            )

    @staticmethod
    def read_scores_file(filepath, header=True):
        """ Function that will read in scores csv file as created by pybkb.learn.backends.gobnilp.write_scores_to_file().
        
        Args:
            :param filepath: Path to scores csv file.
            :type filepath: str
            :param header: Whether a header line exists in the scores csv file.
            :type header: bool
        """
        _scores = defaultdict(lambda: defaultdict(dict))
        with open(filepath, 'r') as f_:
            reader = csv.reader(f_)
            for i, row in enumerate(reader):
                # Remove header row
                if i == 0 and header:
                    continue
                data_idx = int(row[0])
                x_state_idx = row[1]
                score = float(row[2])
                pa_set = frozenset([r for r in row[3:]])
                _scores[data_idx][x_state_idx][pa_set] = score
        # Recast nested defaultdicts as normal dicts
        scores = dict()
        for data_idx, i_scores in _scores.items():
            scores[data_idx] = dict(i_scores)
        return scores
    
    @staticmethod
    def convert_dags_to_bkf(
            dags,
            scores,
            name,
            data,
            feature_states,
            feature_states_index_map,
            store
            ):
        """ Converts DAGs learned by Gobnilp to a BKF inference fragment.

        Args:
            :param dags: A list of dags learned from Gobnilp.
            :type dags: list
            :param scores: Associated scores of learned dags.
            :type scores: list
            :param name: Name of BKF (usually data row index).
            :type name: str
            :param data: Full database that was learned over.
            :type data: np.array
            :param feature_states: List of feature instantiations.
            :type feature_states: list
            :param feature_states_index_map: A dictionary mapping feature state tuples to appropriate column index in the data matrix.
            :type feature_states_index_map: dict
            :param store: Probability store.
            :type store: dict
        """
        bkfs = []
        # Get the minimum score 
        min_score = min(scores)
        # Collect all minimum score dag indices
        dag_indices = [i for i, score in enumerate(scores) if score == min_score]
        for dag_idx in dag_indices:
            bkf = BKB(name)
            dag = dags[dag_idx]
            for node_idx_str in dag.nodes:
                # Gobnilp names are strings so need recast as int
                node_idx = int(node_idx_str)
                # Get head feature name and state name
                head_feature, head_state = feature_states[node_idx]
                # Add to BKF
                bkf.add_inode(head_feature, head_state)
                # Collect all incident nodes to build the tail
                tail_indices = []
                tail = []
                for edge in dag.in_edges(node_idx_str):
                    tail_node_idx = int(edge[0])
                    # Get tail feature and state name
                    tail_feature, tail_state = feature_states[tail_node_idx]
                    # Add to BKF
                    bkf.add_inode(tail_feature, tail_state)
                    # Collect tail
                    tail.append((tail_feature, tail_state))
                    tail_indices.append(tail_node_idx)
                # Calculate S-node conditional probability
                prob, store = conditional_prob(data, node_idx, tail_indices, store)
                # Make S-node
                bkf.add_snode(head_feature, head_state, prob, tail)
            bkfs.append(bkf)
        return bkfs
