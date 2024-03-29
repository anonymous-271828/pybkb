import tqdm
from collections import defaultdict

from pybkb.bkb import BKB


def fuse(bkfs:list, reliabilities:list, source_names:list=None, verbose:bool=False, collapse:bool=True):
    """ Bayesian Knowledge Base Fusion.

    Args:
        :param bkfs: List of BKBs.
        :type bkfs: list
        :param relabilities: List of reliabilites scores for each BKF (float).
        :type reliabilities: list

    Kwargs:
        :param source_names: List of source names for each BKF (str), if None 
            will take the name attribute of each BKF.
        :type source_name: list
        :param verbose: Whether or not to print progress bars.
        :type verbose: bool
        :param collapse: Whether to collapse fused BKB.
        :type: bool
    
    Returns:
        :returns: A fused BKB.
        :rtype: pybkb.bkb.BKB
    """
    # Intialize final fused bkb
    bkb = BKB('fused')
    # Keep track of source component S-nodes so we can normalize later
    source_snodes_by_component = defaultdict(list)
    # Keep track of source I-nodes that we already processed prior S-node
    source_priors_processed = set()
    if source_names is None:
        source_names = [bkf.name for bkf in bkfs]
    # Extract I-nodes
    inodes = set()
    for bkf, reliab, src in tqdm.tqdm(zip(bkfs, reliabilities, source_names), desc='Extracting I-nodes', disable=not verbose, total=len(bkfs)):
        inodes.update(set(bkf.inodes))
    # Make an I-node map
    inodes = list(inodes)
    inode_map = {inode: idx for idx, inode in enumerate(inodes)}
    # Extract all S-Nodes
    all_snodes = defaultdict(list)
    for bkf, reliab, src in tqdm.tqdm(zip(bkfs, reliabilities, source_names), desc='Extracting S-nodes', disable=not verbose, total=len(bkfs)):
        for snode_idx, snode_prob in enumerate(bkf.snode_probs):
            head = bkf.get_snode_head(snode_idx)
            tail = frozenset(bkf.get_snode_tail(snode_idx))
            src_feature = f'__Source__[{head[0]}]'
            all_snodes[(head, tail, snode_prob)].append((reliab, (src_feature, src)))
    # Build Fused BKB
    if collapse:
        return build_fused_collapsed_bkb(inodes, inode_map, all_snodes, verbose)
    return build_fused_bkb(inodes, inode_map, all_snodes, verbose)

def build_fused_bkb(inodes, inode_map, all_snodes, verbose):
    bkb = BKB('fused')
    # Keep track of source component S-nodes so we can normalize later
    source_snodes_by_component = defaultdict(list)
    # Keep track of source I-nodes that we already processed prior S-node
    source_priors_processed = set()
    # Add all I-nodes
    for inode in inodes:
        bkb.add_inode(*inode)
    # Add all source I-nodes
    for _, snode_src_inodes in all_snodes.items():
        for _, src_inode in snode_src_inodes:
            bkb.add_inode(*src_inode)
    # Add all S-nodes
    for (head, tail, prob), snode_src_inodes in tqdm.tqdm(all_snodes.items(), desc='Adding all S-nodes', disable=not verbose):
        for reliab, src_inode in snode_src_inodes:
            # Add source I-node to tail
            tail_with_src = list(tail) + [src_inode]
            bkb.add_snode(head[0], head[1], prob, tail_with_src)
            # Add source prior
            if src_inode not in source_priors_processed:
                source_snodes_by_component[src_inode[0]].append(
                        bkb.add_snode(src_inode[0], src_inode[1], reliab, ignore_prob=True)
                        )
                source_priors_processed.add(src_inode)
    return normalize(bkb, source_snodes_by_component, verbose)

def build_fused_collapsed_bkb(inodes, inode_map, all_snodes, verbose):
    bkb = BKB('fused')
    # Keep track of source component S-nodes so we can normalize later
    source_snodes_by_component = defaultdict(list)
    # Keep track of source I-nodes that we already processed prior S-node
    source_priors_processed = set()
    # Add all I-nodes
    for inode in inodes:
        bkb.add_inode(*inode)
    # Add all source I-node collections
    all_snodes_with_collections = {}
    for snode, snode_src_inodes in all_snodes.items():
        reliab_total = 0
        src_collection_state = []
        for reliab, src_inode in snode_src_inodes:
            src_feature = src_inode[0]
            reliab_total += reliab
            src_collection_state.append(src_inode[1])
        # Add source collection I-node
        src_collection_state = frozenset(src_collection_state)
        bkb.add_inode(src_feature, src_collection_state)
        # Remake all snodes dictionary 
        all_snodes_with_collections[snode] = (reliab_total, (src_feature, src_collection_state))
    # Add all S-nodes
    for (head, tail, prob), src_collection_inode in tqdm.tqdm(all_snodes_with_collections.items(), desc='Adding all S-nodes', disable=not verbose):
        reliab, (src_feature, src_state) = src_collection_inode
        # Add tail
        tail = list(tail) + [(src_feature, src_state)]
        # Add S-node
        bkb.add_snode(head[0], head[1], prob, tail)
        # Add source prior
        if (src_feature, src_state) not in source_priors_processed:
            source_snodes_by_component[src_feature].append(
                    bkb.add_snode(src_feature, src_state, reliab, ignore_prob=True)
                    )
            source_priors_processed.add((src_feature, src_state))
    return normalize(bkb, source_snodes_by_component, verbose)

def normalize(bkb, source_snodes_by_component, verbose):
    # Now Normalize
    for src_feature, snode_indices in tqdm.tqdm(source_snodes_by_component.items(), desc='Normalizing', disable=not verbose):
        # Get the total over each src feature
        total = 0
        for snode_idx in snode_indices:
            total += bkb.snode_probs[snode_idx]
        # Normalize the src feature
        for snode_idx in snode_indices:
            bkb.snode_probs[snode_idx] /= total
    return bkb