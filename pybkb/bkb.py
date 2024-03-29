"""
Primary BKB classes.
"""
import json
import uuid
import compress_pickle
import itertools
import numpy as np
from scipy.sparse import dok_array
from collections import defaultdict
from tqdm import tqdm

from pybkb.exceptions import *
from pybkb.scores import *
from pybkb.utils.probability import build_probability_store
from pybkb.utils import make_hash_sha256
from pybkb.mixins.networkx import BKBNetworkXMixin


class BKB(BKBNetworkXMixin):
    """
    Represents a Bayesian Knowledge Base (BKB) with methods for constructing and manipulating the structure.

    The BKB is composed of instantiation nodes (I-nodes) that represent possible states of components in the knowledge base, and support nodes (S-nodes) that represent probabilistic dependencies between these states.

    :param name: Optional. A name for the BKB. If not provided, a UUID is generated.
    :param description: Optional. A description of the BKB's purpose or contents.
    :param sparse_array_format: The format for representing sparse arrays within the BKB. Defaults to 'dok' (Dictionary Of Keys format). Currently, only 'dok' is implemented.
    :type name: str, optional
    :type description: str, optional
    :type sparse_array_format: str, optional
    :raises ValueError: If an unsupported sparse array format is specified.
    """

    def __init__(self, name: str = None, description: str = None, sparse_array_format: str = 'dok') -> None:
        """
        Initializes a new Bayesian Knowledge Base (BKB) with optional configuration for its name, description, and the format for sparse arrays.

        This method sets up the initial structure of the BKB, including the adjacency matrices for I-nodes and S-nodes and various mappings and lists for managing these nodes.
        """
        if name is None:
            name = str(uuid.uuid4())  # Assign a unique identifier if no name is provided.
        self.name = name
        self.description = description  # Description of the BKB, useful for documentation purposes.

        # Initialize structures for managing I-nodes and S-nodes within the BKB.
        self.head_adj = None  # Adjacency matrix for S-node heads.
        self.tail_adj = None  # Adjacency matrix for S-node tails.
        self.inodes = None  # List of I-nodes (instantiation nodes).
        self.inodes_map = defaultdict(list)  # Maps component names to their states.
        self.inodes_indices_map = {}  # Maps I-nodes to their indices for quick lookup.
        self.snode_probs = []  # List of probabilities associated with each S-node.
        self._snodes = []  # Internal representation of S-nodes for efficient access.
        self.sparse_array_format = sparse_array_format # Set the sparse array representation format.

        # Set the sparse array representation format.
        if sparse_array_format == 'dok':
            self.sparse_array_obj = dok_array  # Use DOK format by default for sparse arrays.
        else:
            raise ValueError(f"Unsupported sparse array format '{sparse_array_format}'. Only 'dok' is currently implemented.")

    def re_annotate_features(self, annotation_map, append=True):
        """ Will append or replace RV names with new names based on annotation_map.
        """
        for old_rv_name, new_rv_name in annotation_map.items():
            # Gather all inode indices that have feature equalling old name
            inode_indices_to_update = []
            for state in self.inodes_map[old_rv_name]:
                inode_indices_to_update.append(self.inodes_indices_map[(old_rv_name, state)])
            # Create new feature name
            if append:
                new_name = f'{old_rv_name}-{new_rv_name}'
            else:
                new_name = new_rv_name
            # Update name in inodes states map
            self.inodes_map[new_name] = self.inodes_map[old_rv_name]
            # Remove the old name
            self.inodes_map.pop(old_rv_name)
            # Update each index
            for inode_idx in inode_indices_to_update:
                _, state = self.inodes[inode_idx]
                new_inode = (new_name, state)
                # Update name in inodes list
                self.inodes[inode_idx] = new_inode
                # Update the inodes indices map
                self.inodes_indices_map[new_inode] = inode_idx

    def save(self, filepath: str, compression: str = 'lz4') -> None:
            """
            Saves the BKB to a file using compressible serialization.

            This method serializes the current state of the BKB into a compressed file, which can be reloaded later.

            :param filepath: The file path where the BKB should be saved.
            :param compression: The compression method to use. Defaults to 'lz4' for fast compression.
            :type filepath: str
            :type compression: str
            """
            with open(filepath, 'wb') as bkb_file:
                compress_pickle.dump(self.dumps(), bkb_file, compression=compression)

    def dumps(self, to_numpy: bool = False):
            """
            Serializes the BKB into a tuple of its constituent components.

            Optionally converts adjacency matrices to dense NumPy arrays for serialization.

            :param to_numpy: If True, converts adjacency matrices to dense NumPy arrays. Defaults to False.
            :type to_numpy: bool
            :return: A tuple containing the serialized components of the BKB.
            :rtype: tuple
            """
            if to_numpy:
                return (self.name, self.description, self.head_adj.toarray(), self.tail_adj.toarray(), self.inodes, np.array(self.snode_probs))
            return (self.name, self.description, self.head_adj, self.tail_adj, self.inodes, self.snode_probs)

    def _rebuild_internal_structures(self):
        """
        Rebuilds internal mappings and structures from the deserialized data.

        This method reconstructs the `inodes_map`, `inodes_indices_map`, and S-nodes representations
        to ensure efficient operation of the BKB object after loading. It is called during the
        deserialization process.
        """
        # Rebuild inodes_map and inodes_indices_map based on loaded inodes
        self.inodes_map = defaultdict(list)
        self.inodes_indices_map = {}
        for idx, (component_name, state_name) in enumerate(self.inodes):
            self.inodes_map[component_name].append(state_name)
            self.inodes_indices_map[(component_name, state_name)] = idx
        
        # Reconstruct the _snodes list for efficient S-node access
        self._snodes = []
        for snode_index, snode_prob in enumerate(self.snode_probs):
            # Determine the head I-node of the S-node
            head_idx = self.head_adj[:, [snode_index]].nonzero()[0][0]
            # Identify tail I-nodes for the S-node
            tail_indices = self.tail_adj[[snode_index], :].nonzero()[1]
            # Add the S-node to the _snodes list with its head index, probability, and tail indices
            self._snodes.append((head_idx, snode_prob, tail_indices)) 

    @classmethod
    def loads(cls, bkb_obj, sparse_array_format: str = 'dok'):
        """
        Deserializes a BKB from its serialized tuple form into a BKB object.

        This class method reconstructs a BKB object from the tuple produced by the `dumps` method.

        :param bkb_obj: The serialized tuple containing the BKB components.
        :param sparse_array_format: The format of sparse arrays to be used in the deserialized BKB. Defaults to 'dok'.
        :type bkb_obj: tuple
        :type sparse_array_format: str
        :return: A BKB object reconstructed from the serialized data.
        :rtype: BKB
        """
        # Initialize an empty BKB with provided name and description
        name, description = bkb_obj[:2]
        bkb = cls(name=name, description=description, sparse_array_format=sparse_array_format)
        
        # Reconstruct the adjacency matrices and node lists from the serialized object
        bkb.head_adj, bkb.tail_adj, bkb.inodes, snode_probs = bkb_obj[2:]
        bkb.snode_probs = list(snode_probs)
        
        # Rebuild internal mappings for quick node lookups
        bkb._rebuild_internal_structures()
        return bkb

    @classmethod
    def load(cls, filepath: str, compression: str = 'lz4', sparse_array_format: str = 'dok'):
        """
        Loads a BKB from a file created by the `save` method.

        This class method deserializes a BKB object from a file, reconstructing its state.

        :param filepath: The file path from which to load the BKB.
        :param compression: The compression method used when the file was saved. Defaults to 'lz4'.
        :param sparse_array_format: The format of sparse arrays in the loaded BKB. Defaults to 'dok'.
        :type filepath: str
        :type compression: str
        :type sparse_array_format: str
        :return: A deserialized BKB object.
        :rtype: BKB
        """
        with open(filepath, 'rb') as bkb_file:
            bkb_tuple = compress_pickle.load(bkb_file, compression=compression)
        return cls.loads(bkb_tuple, sparse_array_format=sparse_array_format)
    
    def get_snode_head(self, snode_index: int) -> tuple:
        """
        Retrieves the head I-node of a specified S-node.

        The head of an S-node is the I-node that the S-node is asserting a probability for, given its tail.

        :param snode_index: The index of the S-node whose head is to be retrieved.
        :type snode_index: int
        :return: A tuple representing the head I-node, in the form (component_name, state_name).
        :rtype: tuple
        """
        head_idx = self._snodes[snode_index][0]
        return self.inodes[head_idx]

    def get_snode_tail(self, snode_index: int) -> list:
        """
        Retrieves the tail I-nodes of a specified S-node.

        The tail of an S-node consists of I-nodes that, when instantiated, influence the probability of the head I-node's state.

        :param snode_index: The index of the S-node whose tail is to be retrieved.
        :type snode_index: int
        :return: A list of tuples representing the tail I-nodes, each in the form (component_name, state_name).
        :rtype: list
        """
        tail_indices = self._snodes[snode_index][2]
        return [self.inodes[tail_idx] for tail_idx in tail_indices]

    def score(
            self,
            data:np.array,
            feature_states:list,
            score_name:str,
            feature_states_index_map:dict=None,
            only:str=None,
            store:dict=None,
            is_learned:bool=False,
            ):
        """ Will score the BKB based on the passed score name.
        
        Args:
            :param data: Full database to learn over.
            :type data: np.array
            :param feature_states: List of feature instantiations.
            :type feature_states: list
            :param score: The name of the score to use: [mdl_ent, mdl_mi].
            :type str:

        Kwargs:
            :param feature_states_index_map: A dictionary mapping feature state tuples to appropriate column index in the data matrix.
            :type feature_states_index_map: dict
            :param only: Return only the data score or model score or both. Options: data, model, both, None. Defaults to None which means both.
            :type only: str
            :param store: A store database of calculated joint probabilties.
            :type store: dict
            :param is_learned: If this is a learned BKB from data then we can use a different MDL calculation
                since each learned data instance BKF corresponds to an inference, i.e. node encoding length is just
                log_2(num_features) instead of log_2(num_inodes) as each BKF will have at most one instantiation of a feature.
            :type is_learned: bool

        """
        # Initalize
        if store is None:
            store = build_probability_store()
        if feature_states_index_map is None:
            # Build feature states index map
            feature_states_index_map = {fs: idx for idx, fs in enumerate(feature_states)}
        # Get score function
        if score_name == 'mdl_ent':
            score_node_obj = MdlEntScoreNode
        elif score_name == 'mdl_mi':
            score_node_obj = MdlMutInfoScoreNode
        else:
            raise ValueError('Unknown score name.')
        # Build all score nodes from each S-node
        score_nodes = []
        if is_learned:
            node_encoding_len = np.log2(len(self.non_source_features))
        else:
            node_encoding_len = np.log2(len(self.non_source_inodes))
        # Collect score nodes and S-nodes by inferences
        snodes_by_inferences = defaultdict(list)
        for snode_idx in range(len(self.snode_probs)):
            head = self.get_snode_head(snode_idx)
            if '__Source__' in head[0]:
                continue
            tail = self.get_snode_tail(snode_idx)
            # Capture the number of sources supporting the S-node
            if is_learned:
                src_inode = [tail_inode for tail_inode in tail if '__Source__' in tail_inode[0]][0]
                src_feature, src_collection = src_inode
                snodes_by_inferences[snode_idx].extend(list(src_collection))
            # Or if it is not a learned bkb or just an inference bkb
            else:
                snodes_by_inferences[snode_idx].append(None)
            # Remove source nodes from tail
            tail = [tail_inode for tail_inode in tail if '__Source__' not in tail_inode[0]]
            score_nodes.append(
                    score_node_obj(head, node_encoding_len, pa_set=tail, indices=False)
                    )
        # Calculate Scores
        model_score = 0
        data_score = 0
        for snode_idx, node in enumerate(score_nodes):
            _dscore, _mscore, _ = node.calculate_score(
                    data,
                    feature_states,
                    store,
                    feature_states_index_map,
                    only='both',
                    )
            model_score += _mscore
            # Multiply data score by number of learned inferences contain the snodes
            data_score += (_dscore * len(snodes_by_inferences[snode_idx]))
        #print(len(snodes_by_inferences))
        if only is None:
            return model_score + data_score
        if only == 'data':
            return data_score
        if only == 'model':
            return model_score
        if only == 'both':
            return data_score, model_score
        raise ValueError('Unknown value for only')

    def _snode_to_dict(self, snode_index: int, prob: float, make_json_serializable: bool = True) -> dict:
        """
        Helper method to convert a single S-node to a dictionary representation.

        :param snode_index: The index of the S-node in the BKB.
        :param prob: The probability of the S-node.
        :param make_json_serializable: If True, converts non-serializable elements (e.g., frozenset) to a serializable format.
        :return: A dictionary representing the S-node, including its head, tail, and probability.
        :rtype: dict
        """
        snode_dict = {"Probability": prob}
        head = self.get_snode_head(snode_index)
        tail = self.get_snode_tail(snode_index)

        snode_dict["Head"] = {head[0]: head[1]}
        snode_dict["Tail"] = {feature: state for feature, state in tail}

        if make_json_serializable:
            for key in ("Head", "Tail"):
                for feature, state in snode_dict[key].items():
                    if isinstance(state, frozenset):
                        snode_dict[key][feature] = str(set(state))  # Convert frozenset to set and then to string

        return snode_dict
    
    def _snodes_to_list(self, make_json_serializable: bool = True) -> list:
        """
        Helper method to convert S-nodes to a list of dictionaries, each representing an S-node.

        :param make_json_serializable: Ensures all elements are JSON serializable by converting incompatible types to strings. Defaults to True.
        :return: A list of dictionaries, each representing an S-node with its head, tail, and probability.
        :rtype: list
        """
        snodes_list = []
        for snode_index in range(len(self.snode_probs)):
            snode_dict = self._snode_to_dict(snode_index, self.snode_probs[snode_index], make_json_serializable)
            snodes_list.append(snode_dict)
        return snodes_list

    def to_dict(self, make_json_serializable: bool = True) -> dict:
        """
        Converts the BKB to a dictionary representation, suitable for serialization or conversion to JSON.

        :param make_json_serializable: If True, ensures that all elements of the dictionary are JSON serializable by converting incompatible types (e.g., sets) to strings. Defaults to True.
        :return: A dictionary representation of the BKB, including its name, description, instantiation nodes (I-nodes), and support nodes (S-nodes) along with their probabilities.
        :rtype: dict
        """
        # Convert S-nodes to list of dictionaries for easy JSON serialization
        snodes_list = self._snodes_to_list(make_json_serializable)
        
        # Prepare instantiation nodes (I-nodes) for serialization
        inodes = {
            feature: [str(state) if make_json_serializable and isinstance(state, frozenset) else state
                      for state in states]
            for feature, states in self.inodes_map.items()
        }

        return {
            "Name": self.name,
            "Description": self.description,
            "Instantiation Nodes": inodes,
            "Support Nodes": snodes_list,
        }

    def json(self, indent: int = 2) -> str:
        """
        Converts the BKB to a JSON string, using the dictionary representation from `to_dict`.

        :param indent: The indentation level for the JSON output. Defaults to 2 for pretty printing.
        :return: A JSON string representation of the BKB.
        :rtype: str
        """
        bkb_dict = self.to_dict()
        return json.dumps(bkb_dict, indent=indent)

    def save_to_json(self, file_path, indent=4):
            """
            Serializes the BKB instance to a JSON file.

            :param file_path: The path to the file where the JSON will be saved.
            :type file_path: str
            """
            bkb_dict = self.to_dict()
            with open(file_path, 'w') as json_file:
                json.dump(bkb_dict, json_file, indent=indent)

    @classmethod
    def load_from_json(cls, file_path):
        """
        Loads a BKB from a JSON file.

        :param file_path: Path to the JSON file containing the BKB data.
        :type file_path: str
        :return: A new instance of BKB populated with the data from the JSON file.
        :rtype: BKB
        """
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        # Assuming the JSON structure directly mirrors the BKB structure
        bkb = cls(data.get('Name'), data.get('Description', None))
        
        for component, states in data.get('Instantiation Nodes', {}).items():
            for state in states:
                bkb.add_inode(component, state)
        
        for snode in data.get('Support Nodes', []):
            head = snode['Head']
            head_component, head_state = list(head.items())[0]
            prob = snode['Probability']
            tail = []
            for tail_component, tail_state in snode.get('Tail', {}).items():
                tail.append((tail_component, tail_state))
            bkb.add_snode(head_component, head_state, prob, tail)
        
        return bkb

    @property
    def source_inodes(self):
        """
        A list of source I-nodes in the BKB.

        Source I-nodes are identified by the presence of '__Source__' in their feature name.

        :return: A list of tuples representing source I-nodes, each tuple in the form (feature, state).
        :rtype: list
        """
        return [(feature, state) for feature, states in self.inodes_map.items() if '__Source__' in feature for state in states]
    
    @property
    def non_source_inodes(self):
        """
        A list of non-source I-nodes in the BKB.

        Non-source I-nodes are identified by the absence of '__Source__' in their feature name.

        :return: A list of tuples representing non-source I-nodes, each tuple in the form (feature, state).
        :rtype: list
        """
        return [(feature, state) for feature, states in self.inodes_map.items() if '__Source__' not in feature for state in states]

    @property
    def non_source_features(self):
        """
        A list of features in the BKB that are not source features.

        Non-source features are identified by the absence of '__Source__' in their name.

        :return: A list of feature names that are not associated with source I-nodes.
        :rtype: list
        """
        return [feature for feature in self.inodes_map if '__Source__' not in feature]
    
    @property
    def source_features(self):
        """
        A list of source features in the BKB.

        Source features are identified by the presence of '__Source__' in their name, indicating that they are associated with source I-nodes.

        :return: A list of feature names that are considered source features.
        :rtype: list
        """
        return [feature for feature in self.inodes_map if '__Source__' in feature]
    
    def _ensure_adjacency_matrices_initialized(self):
        """
        Ensures that the head and tail adjacency matrices are initialized. If they are not,
        initializes them as empty DOK matrices.
        """
        if self.head_adj is None:
            self.head_adj = self.sparse_array_obj((0, 0), dtype=float)
        if self.tail_adj is None:
            self.tail_adj = self.sparse_array_obj((0, 0), dtype=float)

    def _add_inode_to_adj(self) -> None:
        """
        Expands the BKB's adjacency matrices to accommodate a new I-node.

        This internal method adds a row to the head adjacency matrix and a column to the tail
        adjacency matrix, ensuring space for the new I-node's connections.
        """
        self._ensure_adjacency_matrices_initialized()
        # Expand head adjacency by adding a row.
        self.head_adj.resize((self.head_adj.shape[0] + 1, self.head_adj.shape[1]))
        # Expand tail adjacency by adding a column.
        self.tail_adj.resize((self.tail_adj.shape[0], self.tail_adj.shape[1] + 1))

    def _get_inode_index(self, component_name: str, state_name) -> int:
        """
        Retrieves the index of an I-node within the adjacency matrix.

        This method translates the insertion index of an I-node to its actual index in the adjacency matrix.

        :param component_name: The name of the I-node's component (feature).
        :param state_name: The name of the I-node's state (feature state).
        :return: The index of the I-node within the adjacency matrix.
        :raises NoINodeError: If the specified I-node does not exist.
        """
        inode_index = self.inodes_indices_map.get((component_name, state_name))
        if inode_index is None:
            raise NoINodeError(component_name, state_name)
        return inode_index
    
    def add_inode(self, component_name: str, state_name) -> None:
        """
        Adds an I-node to the BKB.

        If the I-node (component_name, state_name) already exists, this method does nothing.
        Otherwise, it adds the new I-node and updates the adjacency matrices to accommodate it.

        :param component_name: The name of the I-node's component (feature).
        :param state_name: The name of the I-node's state (feature state).
        """
        inode_tup = (component_name, state_name)
        # If I-node already exists, do nothing.
        if state_name in self.inodes_map.get(component_name, []):
            return
        # Otherwise, add the new I-node.
        self.inodes = self.inodes or []
        self.inodes.append(inode_tup)
        self.inodes_map[component_name].append(state_name)
        self.inodes_indices_map[inode_tup] = len(self.inodes) - 1
        # Update adjacency matrices to accommodate the new I-node.
        self._add_inode_to_adj()
    
    def _check_snode(self, target_component_name: str, target_state_name, prob: float, tail: list, ignore_prob: bool = False) -> tuple:
        """
        Checks an S-node for validity and translates I-node names into indices.

        Validates the probability value (unless ignored), retrieves the index for the head I-node, and 
        translates tail I-node names into indices.

        :param target_component_name: Name of the S-node's head I-node's component.
        :param target_state_name: State of the S-node's head I-node.
        :param prob: Probability associated with the S-node.
        :param tail: List of tuples representing the S-node's tail I-nodes.
        :param ignore_prob: If True, does not validate the probability value.
        :return: A tuple containing the head index, probability, and list of tail indices.
        :raises SNodeProbError: If the probability is invalid and not ignored.
        """
        if not ignore_prob and not (0 <= prob <= 1):
            raise SNodeProbError(prob)

        head_idx = self._get_inode_index(target_component_name, target_state_name)
        tail_indices = [self._get_inode_index(comp_name, state_name) for comp_name, state_name in tail] if tail else []

        return head_idx, prob, tail_indices

    def add_snode(self, target_component_name: str, target_state_name, prob: float, tail: list = None, ignore_prob: bool = False) -> int:
        """
        Adds a Support Node (S-node) to the BKB.

        :param target_component_name: Component name of the S-node's head I-node.
        :param target_state_name: State name of the S-node's head I-node.
        :param prob: Probability associated with the S-node.
        :param tail: Optional list of tuples representing the tail I-nodes.
        :param ignore_prob: If True, allows probabilities outside [0,1].
        """
        head_idx, prob, tail_indices = self._check_snode(target_component_name, target_state_name, prob, tail or [], ignore_prob)

        self._ensure_adjacency_matrices_initialized()
        num_snodes = len(self.snode_probs)
        self.head_adj.resize((len(self.inodes), num_snodes + 1))
        self.tail_adj.resize((num_snodes + 1, len(self.inodes)))

        for tail_idx in tail_indices:
            self.tail_adj[num_snodes, tail_idx] = 1
        self.head_adj[head_idx, num_snodes] = 1

        self.snode_probs.append(prob)
        self._snodes.append((head_idx, prob, tail_indices))
        # Return the S-node id
        return len(self.snode_probs) - 1

    def find_snodes(self, target_component_name: str, target_state_name, prob: float = None, tail_subset: list = None):
        """
        Finds S-nodes matching specified head component/state, probability, and tail subset criteria.

        :param target_component_name: Name of the head I-node's component.
        :param target_state_name: State of the head I-node.
        :param prob: Optional. Probability associated with the S-node to filter by.
        :param tail_subset: Optional. A list of tail I-node (component, state) tuples to filter by.
        :return: Indices of S-nodes in the BKB that match all specified criteria.
        """
        head_idx = self._get_inode_index(target_component_name, target_state_name)
        # Find all S-nodes with the specified head.
        snode_indices = self.head_adj[[head_idx], :].nonzero()[1]

        # Apply filters for probability and tail subset if specified.
        filtered_indices = []
        for snode_idx in snode_indices:
            if prob is not None and self.snode_probs[snode_idx] != prob:
                continue  # Skip S-nodes not matching the probability if specified.

            tail = self.get_snode_tail(snode_idx)
            tail_set = set(tail)
            
            if tail_subset:
                tail_subset_set = set(tail_subset)
                # Check if the specified tail subset matches the S-node's tail.
                if not tail_subset_set.issubset(tail_set):
                    continue  # Skip S-nodes not matching the tail subset if specified.
            
            # If all filters pass, add the S-node index to the result.
            filtered_indices.append(snode_idx)
            
        return filtered_indices

    @property
    def snodes_by_head(self):
        """
        Groups S-nodes by their head I-node.

        This property creates a dictionary where each key is a tuple representing a head I-node,
        and the value is a list of indices of S-nodes that have this I-node as their head.

        :return: A dictionary mapping head I-nodes to lists of S-node indices.
        :rtype: dict[tuple, list[int]]
        """
        snodes_by_head = defaultdict(list)
        for snode_idx in range(len(self.snode_probs)):
            head = self.get_snode_head(snode_idx)
            snodes_by_head[head].append(snode_idx)
        return dict(snodes_by_head)
   
    @property
    def snodes_by_tail(self):
        """
        Groups S-nodes by their tail I-nodes.

        This property creates a dictionary where each key is a frozenset representing the tail I-nodes,
        and the value is a list of indices of S-nodes that have these I-nodes in their tail.

        :return: A dictionary mapping sets of tail I-nodes to lists of S-node indices. The use of frozenset ensures that the order of tail I-nodes does not affect the grouping.
        :rtype: dict[frozenset, list[int]]
        """
        snodes_by_tail = defaultdict(list)
        for snode_idx in range(len(self.snode_probs)):
            tail = frozenset(self.get_snode_tail(snode_idx))
            snodes_by_tail[tail].append(snode_idx)
        return dict(snodes_by_tail)

    def are_snodes_mutex(self, snode_index1, snode_index2, check_head=True):
        """
        Determines if two S-nodes are mutually exclusive based on their heads and tails.

        :param snode_index1: Index of the first S-node.
        :param snode_index2: Index of the second S-node.
        :param check_head: Whether to check if the S-nodes have different heads as part of the mutex check.
        :return: True if the S-nodes are mutually exclusive, False otherwise.
        """
        if check_head and self.get_snode_head(snode_index1) != self.get_snode_head(snode_index2):
            return True  # S-nodes with different heads are considered mutually exclusive.

        tail1, tail2 = map(set, (self.get_snode_tail(snode_index1), self.get_snode_tail(snode_index2)))

        # Mutual exclusion is violated if both tails are identical.
        return tail1 != tail2

    def is_mutex(self, verbose=False) -> bool:
        """
        Checks the entire BKB to ensure that all S-nodes are mutually exclusive where required.

        :param verbose: If True, outputs progress and diagnostic information.
        :return: True if the mutual exclusion condition holds across the BKB, False if violated.
        :raises BKBNotMutexError: If any pair of S-nodes violates the mutual exclusion condition.
        """
        for adj_idx in tqdm.tqdm(range(len(self.inodes)), desc='Checking Head I-nodes', disable=not verbose):
            snode_indices = self.head_adj[[adj_idx], :].nonzero()[1]

            for snode_idx1, snode_idx2 in itertools.combinations(snode_indices, 2):
                if not self.are_snodes_mutex(snode_idx1, snode_idx2, check_head=False):
                    if verbose:
                        tqdm.write(f'Mutex violation between S-node {snode_idx1} and S-node {snode_idx2}')
                    raise BKBNotMutexError(snode_idx1, snode_idx2)

        return True
    @classmethod
    def union(cls, *bkbs, sparse_array_format='dok'):
        """
        Creates a union of multiple BKBs into a single BKB. This method combines the I-nodes and S-nodes from the given BKBs.

        Note: This method does not guarantee that the resulting union BKB will be mutually exclusive (Mutex).

        :param bkbs: An arbitrary number of BKB objects to be unified.
        :param sparse_array_format: The format of sparse arrays in the resulting union BKB. Defaults to 'dok'.
        :return: A new BKB instance representing the union of the input BKBs.
        """
        # Construct a unique name for the unioned BKB based on the names of the input BKBs.
        name = 'Union of: ' + ', '.join(bkb.name for bkb in bkbs)

        # Initialize the unioned BKB with the constructed name and specified sparse array format.
        unioned = cls(name=name, sparse_array_format=sparse_array_format)

        # Iterate over each BKB to collect and add unique I-nodes and S-nodes to the unioned BKB.
        for bkb in bkbs:
            # Add unique I-nodes.
            for inode in bkb.inodes:
                if not unioned.inodes or inode not in unioned.inodes:
                    unioned.add_inode(*inode)

            # Add S-nodes.
            for snode_idx, snode_prob in enumerate(bkb.snode_probs):
                snode_head = bkb.get_snode_head(snode_idx)
                snode_tail = bkb.get_snode_tail(snode_idx)
                # Avoid adding duplicate S-nodes by checking if the unioned BKB already contains the S-node.
                if not unioned.find_snodes(snode_head[0], snode_head[1], prob=snode_prob, tail_subset=snode_tail):
                    unioned.add_snode(snode_head[0], snode_head[1], prob=snode_prob, tail=snode_tail)

        return unioned

    def get_causal_ruleset(self) -> dict:
        """
        Maps each feature to a list of indices of S-nodes forming causal rulesets.

        A causal ruleset for a feature consists of all S-nodes where the feature acts as the head
        in the causal relationship represented by the S-node.

        :return: A dictionary where keys are feature names and values are lists of S-node indices.
        """
        causal_rulesets = defaultdict(list)
        for snode_index in range(len(self.snode_probs)):
            head_inode = self.get_snode_head(snode_index)
            feature = head_inode[0]
            causal_rulesets[feature].append(snode_index)

        return dict(causal_rulesets)

    def __eq__(self, other) -> bool:
        """
        Determines if two BKB instances are equal based on their I-nodes and S-nodes.

        :param other: Another BKB instance to compare with.
        :return: True if both instances have the same I-nodes and S-nodes, False otherwise.
        """
        if not isinstance(other, BKB):
            return NotImplemented

        # Direct comparison of I-nodes for equality.
        if set(self.inodes) != set(other.inodes):
            print('Inodes not equal')
            return False

        # Ensuring both BKBs have the same S-node probabilities and structures.
        if len(self.snode_probs) != len(other.snode_probs):
            print('Snode len not equal')
            return False

        for snode_idx, _ in enumerate(self.snode_probs):
            snode_other_indices = other.find_snodes(
                *self.get_snode_head(snode_idx),
                prob=self.snode_probs[snode_idx], 
                tail_subset=self.get_snode_tail(snode_idx)
                )
            # Should only be one S-node with the same head and tail and prob
            if len(snode_other_indices) > 1:
                print('Multiple S-nodes found with same head, tail, and prob')
                return False
            if not snode_other_indices:
                print('Snode not found with same head, tail, and prob')
                return False
            other_snode_idx = snode_other_indices[0]
            if self.snode_probs[snode_idx] != other.snode_probs[other_snode_idx]:
                print('Snode probs not equal for index:', snode_idx)
                return False

        return True
    
    def __hash__(self):
        """
        Generates a hash value for a BKB instance.

        The hash is based on the deterministic serialization of the BKB to a dictionary, ensuring that
        equal BKBs produce the same hash value.

        :return: The hash value of the BKB instance.
        """
        # Ensuring the dictionary representation is sorted to produce a consistent hash value.
        serialized_dict = self.to_dict(make_json_serializable=True)
        sorted_serialized_str = str(sorted(serialized_dict.items()))
        return hash(sorted_serialized_str)

