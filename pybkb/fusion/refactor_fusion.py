import tqdm
from collections import defaultdict

from pybkb.bkb import BKB


def fuse(bkbs, reliabilities, source_names=None, verbose=False, collapse=True):
    """
    Fuses multiple Bayesian Knowledge Bases (BKBs) based on their reliabilities and optional source names.

    :param bkbs: A list of BKB instances to be fused.
    :type bkbs: list[BKB]
    :param reliabilities: A list of reliability scores corresponding to each BKB.
    :type reliabilities: list[float]
    :param source_names: Optional; a list of source names for each BKB. Defaults to BKB names if not provided.
    :type source_names: list[str], optional
    :param verbose: If True, displays progress bars during the fusion process.
    :type verbose: bool
    :param collapse: If True, collapses the fused BKB to improve reasoning efficiency. Defaults to True.
    :type collapse: bool
    :return: A fused BKB instance.
    :rtype: BKB

    This function aggregates and fuses the knowledge contained in multiple BKBs, taking into account the reliability
    of each source to construct a comprehensive and coherent BKB that reflects the combined knowledge.
    """
    source_names = source_names or [bkf.name for bkf in bkbs]

    inodes = _extract_inodes(bkbs, verbose)
    all_snodes = _extract_snodes(bkbs, reliabilities, source_names, verbose)

    if collapse:
        return build_fused_collapsed_bkb(inodes, all_snodes, verbose)
    return build_fused_bkb(inodes, all_snodes, verbose)

def _extract_inodes(bkbs, verbose):
    """
    Extracts a unique set of I-nodes from a list of BKBs.

    :param bkbs: A list of BKB instances.
    :type bkbs: list[BKB]
    :param verbose: Enables progress display if True.
    :type verbose: bool
    :return: A list of unique I-nodes across all BKBs.
    :rtype: list
    """
    inodes = set()
    for bkf in tqdm.tqdm(bkbs, desc='Extracting I-nodes', disable=not verbose):
        inodes.update(bkf.inodes)
    return list(inodes)

def _extract_snodes(bkbs, reliabilities, source_names, verbose):
    """
    Extracts and organizes S-nodes from a list of BKBs, considering their reliabilities and source names.

    :param bkbs: A list of BKB instances.
    :type bkbs: list[BKB]
    :param reliabilities: A list of reliability scores for each BKB.
    :type reliabilities: list[float]
    :param source_names: A list of source names for each BKB.
    :type source_names: list[str]
    :param verbose: Enables progress display if True.
    :type verbose: bool
    :return: A dictionary of S-nodes keyed by their head, tail, and probability, with associated reliabilities and source I-nodes.
    :rtype: defaultdict[list]
    """
    all_snodes = defaultdict(list)
    for bkf, reliability, source_name in tqdm.tqdm(zip(bkbs, reliabilities, source_names), desc='Extracting S-nodes', disable=not verbose, total=len(bkbs)):
        for snode_idx, snode_prob in enumerate(bkf.snode_probs):
            head = bkf.get_snode_head(snode_idx)
            tail = frozenset(bkf.get_snode_tail(snode_idx))
            src_feature = f'__Source__[{head[0]}]'
            all_snodes[(head, tail, snode_prob)].append((reliability, (src_feature, source_name)))
    return all_snodes


def build_fused_bkb(inodes, all_snodes, verbose):
    """
    Constructs a fused Bayesian Knowledge Base (BKB) from aggregated I-nodes and S-nodes.

    :param inodes: A list of all unique I-nodes across the BKBs to be fused.
    :type inodes: list
    :param all_snodes: A dictionary where keys are tuples representing S-nodes (head, tail, probability),
                       and values are lists of tuples containing reliability and source I-node information.
    :type all_snodes: defaultdict(list)
    :param verbose: If True, progress information will be displayed during the fusion process.
    :type verbose: bool
    :return: A fused BKB instance with normalized source component S-nodes.
    :rtype: BKB

    The function adds all I-nodes and S-nodes to a new BKB instance, accounting for source reliability
    and ensuring that S-nodes from different sources are correctly incorporated into the fused BKB.
    """
    bkb = BKB('fused')
    
    # Initialize structures for tracking source component S-nodes and processed source I-nodes.
    source_snodes_by_component = defaultdict(list)
    source_priors_processed = set()
    
    # Add all unique I-nodes to the fused BKB.
    for inode in inodes:
        bkb.add_inode(*inode)
    
    # Process all S-nodes, incorporating source reliability into the fusion.
    for (head, tail, prob), snode_src_inodes in tqdm.tqdm(all_snodes.items(), desc='Adding all S-nodes', disable=not verbose):
        for reliab, src_inode in snode_src_inodes:
            # Add Source I-node and prior
            bkb.add_inode(src_inode[0], src_inode[1])
            # Record and normalize source priors to ensure they contribute appropriately to the fused BKB.
            if src_inode not in source_priors_processed:
                source_snodes_by_component[src_inode[0]].append(
                    bkb.add_snode(src_inode[0], src_inode[1], reliab, ignore_prob=True)
                )
                source_priors_processed.add(src_inode)

            # Extend the S-node's tail with its corresponding source I-node for reliability tracking.
            tail_with_src = list(tail) + [src_inode]
            bkb.add_snode(head[0], head[1], prob, tail_with_src)
                
    # Normalize the fused BKB to adjust source priors and finalize the fusion process.
    return normalize(bkb, source_snodes_by_component, verbose)

def build_fused_collapsed_bkb(inodes, all_snodes, verbose):
    """
    Constructs a fused Bayesian Knowledge Base (BKB) with collapsed source node collections for efficiency.

    This method creates a single BKB from aggregated I-nodes and S-nodes while collapsing source nodes into
    collections. This approach reduces the complexity and improves the efficiency of reasoning over the fused BKB.

    :param inodes: All unique I-nodes across the BKBs being fused.
    :type inodes: list
    :param all_snodes: Aggregated S-node data, including heads, tails, probabilities, and associated source information.
    :type all_snodes: defaultdict
    :param verbose: If set to True, progress updates will be shown.
    :type verbose: bool
    :return: A fused BKB instance with normalized source component S-nodes.
    :rtype: BKB

    The fusion process accounts for the reliability of different sources by creating collections of source nodes,
    allowing for more efficient probabilistic reasoning with the fused knowledge.
    """
    bkb = BKB('fused')
    source_snodes_by_component = defaultdict(list)
    source_priors_processed = set()

    # Add all unique I-nodes to the new BKB instance.
    for inode in inodes:
        bkb.add_inode(*inode)

    # Process S-node source information to collapse source nodes into collections.
    all_snodes_with_collections = _process_source_nodes(all_snodes)

    # Add S-nodes to the fused BKB, incorporating collapsed source node collections.
    for (head, tail, prob), src_collection_data in tqdm.tqdm(all_snodes_with_collections.items(), desc='Adding all S-nodes', disable=not verbose):
        _add_collapsed_snode(bkb, head, tail, prob, src_collection_data, source_snodes_by_component, source_priors_processed)

    return normalize(bkb, source_snodes_by_component, verbose)

def _process_source_nodes(all_snodes):
    """
    Processes S-nodes to aggregate and collapse source node information into collections.

    :param all_snodes: Aggregated S-node data from multiple BKBs.
    :type all_snodes: defaultdict
    :return: A dictionary of processed S-nodes with source node collections.
    :rtype: dict
    """
    all_snodes_with_collections = {}
    for snode, snode_src_inodes in all_snodes.items():
        reliab_total, src_collection_state = _aggregate_source_info(snode_src_inodes)
        all_snodes_with_collections[snode] = (reliab_total, src_collection_state)
    return all_snodes_with_collections

def _aggregate_source_info(snode_src_inodes):
    """
    Aggregates reliability scores and source states for a given S-node.

    :param snode_src_inodes: Source nodes and reliability scores associated with an S-node.
    :type snode_src_inodes: list[tuple]
    :return: Total reliability score and a collection of source states.
    :rtype: tuple
    """
    reliab_total = 0
    src_collection_state = []
    for reliab, src_inode in snode_src_inodes:
        reliab_total += reliab
        src_collection_state.append(src_inode[1])
    src_collection_state = frozenset(src_collection_state)
    return reliab_total, (src_inode[0], src_collection_state)

def _add_collapsed_snode(bkb, head, tail, prob, src_collection_data, source_snodes_by_component, source_priors_processed):
    """
    Adds an S-node to the BKB with collapsed source node information.

    :param bkb: The BKB instance being constructed.
    :param head: The head of the S-node.
    :param tail: The tail of the S-node.
    :param prob: The probability of the S-node.
    :param src_collection_data: Collapsed source node data for the S-node.
    :param source_snodes_by_component: Tracking of source S-nodes for normalization.
    :param source_priors_processed: Set of processed source priors to avoid duplication.
    """
    reliab, (src_feature, src_state) = src_collection_data
    tail_with_src = list(tail) + [(src_feature, src_state)]
    # Add Source Collection I-node
    bkb.add_inode(src_feature, src_state)
    # Add S-node with Source Collection Tail
    bkb.add_snode(head[0], head[1], prob, tail_with_src)

    # Add Source Collection Prior if not already processed
    if (src_feature, src_state) not in source_priors_processed:
        snode_idx = bkb.add_snode(src_feature, src_state, reliab, ignore_prob=True)
        source_snodes_by_component[src_feature].append(snode_idx)
        source_priors_processed.add((src_feature, src_state))

def normalize(bkb, source_snodes_by_component, verbose):
    """
    Normalizes the probabilities of source component S-nodes within a Bayesian Knowledge Base (BKB).

    :param bkb: The Bayesian Knowledge Base instance being normalized. This BKB should already contain S-nodes
                that might have unnormalized probabilities due to the fusion of multiple sources.
    :type bkb: BKB
    :param source_snodes_by_component: A mapping from source component features to lists of indices
                                       corresponding to S-nodes associated with each source component. 
                                       These S-nodes are targeted for probability normalization.
    :type source_snodes_by_component: dict
    :param verbose: If True, displays progress information.
    :type verbose: bool
    :return: The normalized BKB instance.
    :rtype: BKB

    This function adjusts the probabilities of S-nodes related to source components so that the total probability
    for each source component equals 1. This normalization is necessary when S-nodes from different sources
    contribute to the same feature and need to be balanced according to their reliability or frequency.
    """
    # Iterate over each source component to normalize associated S-node probabilities.
    for _, snode_indices in tqdm.tqdm(source_snodes_by_component.items(), desc='Normalizing', disable=not verbose):
        total = sum(bkb.snode_probs[snode_idx] for snode_idx in snode_indices)
        
        # Normalize each S-node's probability by the total for its source component.
        for snode_idx in snode_indices:
            bkb.snode_probs[snode_idx] /= total
            
    return bkb