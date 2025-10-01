import pickle
import random
import math
import re
from scipy.cluster.hierarchy import to_tree
import copy
from thefuzz import fuzz  # pip install thefuzz[speedup]
import spacy  # pip install spacy
# Download model: python -m spacy download en_core_web_md

######################
# binary tree utils
######################
def get_binary_tree(linkage_matrix):
    """get a binary tree from a linkage matrix"""
    root, nodelist = to_tree(linkage_matrix, rd=True)
    return root, nodelist


def find_binary_tree_node_by_id(root, id):
    if root.id == id:
        return root
    if root.left:
        left_node = find_binary_tree_node_by_id(root.left, id)
        if left_node:
            return left_node
    if root.right:
        right_node = find_binary_tree_node_by_id(root.right, id)
        if right_node:
            return right_node
    return None


def assign_leaf_clusters(root, threshold=1.08):
    """cut a binary tree at a threshold and assign labels and datapoints"""
    # Dictionary to store, for each leaf, its assigned cluster representative.
    leaf_to_rep = {}

    # label to level
    label_to_level = {}
    def _assign_leaf_clusters(node, level, current_label):
        """
        Recursively traverse the dendrogram tree to assign cluster labels and levels.
        
        Parameters:
        node         : current node in the tree.
        level        : the level of the node in the tree (starts at 0 for root).
        current_label: the label inherited from merges with a distance <= threshold.
                        
        Behavior:
        - Root node starts at level 0
        - Each cut point increases the level for its children
        - Merged nodes (below threshold) are assigned level -1
        """
        if node.is_leaf():
            # Leaf nodes inherit their parent's level if merged (current_label exists)
            # or keep their own level if they're at a cut point
            node.data_points = [node.id]
            node.level = -1 if current_label is not None else level
            label = current_label if current_label is not None else node.id
            leaf_to_rep[node.id] = label
            return [node.id]

        if node.dist <= threshold:
            # Merged nodes and their children all get level -1
            node.level = level
            new_label = current_label if current_label is not None else node.id
            label_to_level[new_label] = level
            data_points = []
            # Pass level -1 to children as they're part of the merged group
            data_points.extend(_assign_leaf_clusters(node.left, -1, new_label))
            data_points.extend(_assign_leaf_clusters(node.right, -1, new_label))
            node.data_points = data_points
            return data_points
        else:
            # Cut point: assign current level and increment for children
            node.level = level
            data_points = []
            # Increment level for children as this is a cut point
            data_points.extend(_assign_leaf_clusters(node.left, level+1, None))
            data_points.extend(_assign_leaf_clusters(node.right, level+1, None))
            node.data_points = data_points
            return data_points
    _assign_leaf_clusters(root, 0, None)
    return leaf_to_rep, label_to_level


def get_all_nodes_to_label(root):
    """
    Recursively collect all nodes that need labels (nodes with level >= 0).
    These are all nodes except the merged leaf nodes (level -1).
    """
    all_nodes_to_label = []
    def _get_all_nodes_to_label(node):
        if hasattr(node, 'level') and node.level >= 0:
            all_nodes_to_label.append(node)

        if not node.is_leaf():
            _get_all_nodes_to_label(node.left)
            _get_all_nodes_to_label(node.right)

    _get_all_nodes_to_label(root)
    print(f"Number of nodes to label: {len(all_nodes_to_label)}")
    return all_nodes_to_label


def balanced_sampling(root, k):
    """
    Recursively sample data points from the tree, skipping nodes where node.level == -1.
    
    Parameters:
      - root: the root node of the tree.
      - k: target sample size for leaf nodes.
    
    Returns:
      A dictionary mapping node.id to its balanced sampled data points (for nodes with level != -1).
    """
    sampled_dict = {}

    def sample_node(node, k):
        # Safely get the data_points_indices attribute; default to empty list if not present.
        data_points = getattr(node, 'data_points', [])
        if not data_points:
            return []
        if node.level == -1:
            sampled = data_points.copy() if len(data_points) <= k else random.sample(data_points, k)
            # shuffle the sampled data points
            random.shuffle(sampled)
            return sampled
        
        # Internal node: get samples recursively from children.
        left_samples = sample_node(node.left, k) if node.left is not None else []
        right_samples = sample_node(node.right, k) if node.right is not None else []

        if node.left is not None and node.right is not None:
            # Determine balanced sample size per child.
            sample_size_per_child = math.ceil(k / 2)
            left_sampled = left_samples.copy() if len(left_samples) <= sample_size_per_child else random.sample(left_samples, sample_size_per_child)
            right_sampled = right_samples.copy() if len(right_samples) <= sample_size_per_child else random.sample(right_samples, sample_size_per_child)
            combined_samples = left_sampled + right_sampled
        else:
            # If only one child exists, use its samples.
            combined_samples = left_samples if node.left is not None else right_samples
        if not combined_samples:
            combined_samples = data_points.copy() if len(data_points) <= k else random.sample(data_points, k)
        # Store this node's result only if its level is not -1.
        assert node.level != -1
        # shuffle the sampled data points
        random.shuffle(combined_samples)
        sampled_dict[node.id] = combined_samples
        
        return combined_samples

    sample_node(root, k)
    return sampled_dict


def count_nodes_binary(node):
    if node is None:
        return 0
    return 1 + count_nodes_binary(node.left) + count_nodes_binary(node.right)


def assign_annotations(node, annotations):
    """
    Assign the annotations to the node based on its id
    node is a binary tree node
    annotations is a dictionary of annotations {id: {annotation: str, sampled_job_titles: list, sampled_job_idx: list}}
    """
    if node is None:
        return
        
    # Ensure label attribute exists for this node
    setattr(node, 'label', '')
    setattr(node, 'sampled_titles', [])
    setattr(node, 'sampled_job_idx', [])
    # Recursively ensure labels exist for children first
    assign_annotations(node.left, annotations)
    assign_annotations(node.right, annotations)
    
    # Now assign the actual label
    if node.level <= 0:
        node.label = ''
        node.sampled_titles = []
        node.sampled_job_idx = []
    else:
        node.label = annotations[int(node.id)]['annotation']
        node.sampled_titles = annotations[int(node.id)]['sampled_job_titles']
        node.sampled_job_idx = annotations[int(node.id)]['sampled_job_idx']

######################
# n-ary tree utils
######################
class FlexibleTreeNode:
    def __init__(self, binary_node):
        # Copy all attributes from the binary node except 'left' and 'right'
        for attr, value in binary_node.__dict__.items():
            if attr not in ['left', 'right']:
                setattr(self, attr, value)
        self.kids = []  # flexible list of children

def convert_binary_to_flexible(binary_node):
    """
    Recursively convert a binary tree node (with left/right pointers)
    to a FlexibleTreeNode (with a kids list). Preserves all attributes except left/right.
    """
    if binary_node is None:
        return None
    new_node = FlexibleTreeNode(binary_node)
    # Add left and right children (if they exist) into the kids list.
    for child in [binary_node.left, binary_node.right]:
        if child is not None:
            new_child = convert_binary_to_flexible(child)
            new_node.kids.append(new_child)
    return new_node


def find_node_by_id(root, id):
    if root is None:
        return None
    if root.id == id:
        return root
    for kid in root.kids:
        found = find_node_by_id(kid, id)
        if found is not None:
            return found
    return None


def count_nodes_flexible(node):
    if node is None:
        return 0
    return 1 + sum(count_nodes_flexible(child) for child in node.kids)


def copy_flexible_tree(node):
    """
    Recursively creates a deep copy of the flexible tree starting at `node`.
    This ensures that modifications (like pruning) on the copy will not affect the original tree.
    """
    if node is None:
        return None
    # Create a new node using the FlexibleTreeNode constructor with a dummy binary node
    new_node = FlexibleTreeNode(node)
    # Recursively copy each child in the kids list
    new_node.kids = [copy_flexible_tree(child) for child in node.kids]
    return new_node


def trim_trailing_empty(path):
    """
    Return a copy of the path with trailing nodes having an empty label removed.
    This yields the "meaningful" portion of the branch.
    """
    trimmed = list(path)
    while trimmed and trimmed[-1][1] == '':
        trimmed.pop()
    return trimmed


def dfs_longest_repetitions_global_dedup_flexible(node, path, label_count, results, global_occurrences, seen, 
                                                 use_similarity=False, similarity_method='combined', 
                                                 similarity_threshold=0.85):
    """
    DFS that, at every leaf:
      - Updates global_occurrences for each unique meaningful label in the trimmed branch.
      - Records the branch result (with deduplication) if there's any repeated label.
    
    :param node: current tree node.
    :param path: list of (node.id, node.label) tuples along the current branch.
    :param label_count: dict counting occurrences of each meaningful label along the branch.
    :param results: list to collect branch results (only for branches with repetition).
    :param global_occurrences: dict mapping each label to a set of branch identifiers.
    :param seen: set to track unique trimmed branch keys (for deduplication of results).
    :param use_similarity: whether to use similarity matching instead of exact matching.
    :param similarity_method: method to use for similarity comparison ('combined', 'spacy', or 'fuzzy').
    :param similarity_threshold: threshold above which labels are considered similar.
    """
    if node is None:
        return

    # Append current node to the path.
    path.append((node.id, node.label))
    
    if node.label:  # count only meaningful labels
        if not use_similarity:
            # Exact matching (original behavior)
            label_count[node.label] = label_count.get(node.label, 0) + 1
        else:
            # Similarity matching
            # First, check if this label is similar to any existing label
            similar_found = False
            for existing_label in list(label_count.keys()):
                if existing_label == 'similar_mappings':
                    continue
                    
                if labels_are_similar(node.label, existing_label, 
                                     method=similarity_method, 
                                     threshold=similarity_threshold):
                    # Consider them the same label for counting purposes
                    label_count[existing_label] += 1
                    # Store the mapping for backtracking
                    if 'similar_mappings' not in label_count:
                        label_count['similar_mappings'] = {}
                    if node.label not in label_count['similar_mappings']:
                        label_count['similar_mappings'][node.label] = []
                    label_count['similar_mappings'][node.label].append(existing_label)
                    similar_found = True
                    break
            
            # If no similar label found, add this as a new label
            if not similar_found:
                label_count[node.label] = label_count.get(node.label, 0) + 1

    # If this is a leaf, update the global counter and possibly record a result.
    if not node.kids:
        # Compute the trimmed branch (drop trailing nodes with empty labels).
        trimmed = trim_trailing_empty(path)
        key = tuple(trimmed)  # Use the trimmed branch as a unique key.
        branch_id = tuple(nid for nid, lbl in trimmed)  # branch identifier based on node ids.
        
        # Update global_occurrences for every meaningful label in the branch.
        unique_labels = {lbl for nid, lbl in trimmed if lbl}
        
        if not use_similarity:
            # Original exact matching behavior
            for label in unique_labels:
                global_occurrences.setdefault(label, set()).add(branch_id)
            
            # Check for any repeated labels along this branch.
            repeated = {label: count for label, count in label_count.items() 
                       if label and label != 'similar_mappings' and count > 1}
        else:
            # Similarity matching for global occurrences
            for label in unique_labels:
                # Check if this label is similar to any existing label in global_occurrences
                similar_found = False
                for existing_label in list(global_occurrences.keys()):
                    if labels_are_similar(label, existing_label, 
                                         method=similarity_method, 
                                         threshold=similarity_threshold):
                        global_occurrences[existing_label].add(branch_id)
                        similar_found = True
                        break
                
                # If no similar label found, add this as a new label
                if not similar_found:
                    global_occurrences.setdefault(label, set()).add(branch_id)
            
            # Check for any repeated labels along this branch (excluding the mappings dict)
            repeated = {label: count for label, count in label_count.items() 
                       if label and label != 'similar_mappings' and count > 1}
        
        if repeated:
            # Record this branch's result only if not seen before.
            if key not in seen:
                seen.add(key)
                results.append({
                    "path": list(path),  # you can store the full path or the trimmed version
                    "repeated_labels": repeated
                })
    else:
        # Recurse into children.
        for kid in node.kids:
            dfs_longest_repetitions_global_dedup_flexible(kid, path, label_count, results, global_occurrences, seen,
                                                         use_similarity, similarity_method, similarity_threshold)

    # Backtrack: remove the current node and update the count.
    if node.label:
        if not use_similarity:
            # Original exact matching behavior
            label_count[node.label] -= 1
            if label_count[node.label] == 0:
                del label_count[node.label]
        else:
            # Similarity matching backtracking
            if 'similar_mappings' in label_count and node.label in label_count['similar_mappings']:
                # This label was mapped to an existing similar label
                for similar_label in label_count['similar_mappings'][node.label]:
                    label_count[similar_label] -= 1
                    if label_count[similar_label] == 0:
                        del label_count[similar_label]
                # Remove the mapping
                del label_count['similar_mappings'][node.label]
                if not label_count['similar_mappings']:
                    del label_count['similar_mappings']
            else:
                # This was a unique label
                if node.label in label_count:  # Check if the label exists before decrementing
                    label_count[node.label] -= 1
                    if label_count[node.label] == 0:
                        del label_count[node.label]
    
    path.pop()


def find_sibling_repetitions_nary(root, use_similarity=False, similarity_method='combined', similarity_threshold=0.85):
    """
    Recursively finds sibling repetitions in an n-ary tree.
    
    A sibling repetition occurs at a parent node if two or more of its children have the same non-empty label.
    For each such parent, the function records the parent's id along with a list of unique repeated labels.
    
    Args:
        root: The root node of the tree.
        use_similarity: Whether to use similarity matching instead of exact matching.
        similarity_method: Method to use for similarity comparison ('combined', 'spacy', or 'fuzzy').
        similarity_threshold: Threshold above which labels are considered similar.
    
    Returns:
        A list of dictionaries, each containing:
            - "parent": the parent's (id, label) tuple.
            - "kids": list of (id, label) tuples for all children.
            - "repeated_labels": a list of labels that appear two or more times among the parent's children,
                                or a list of dicts with canonical labels and their variants when using similarity.
    """
    results = []
    
    def helper(node):
        if node is None:
            return
        if node.kids:
            if not use_similarity:
                # Original exact matching behavior
                freq = {}
                for child in node.kids:
                    lab = child.label
                    if lab:  # consider only non-empty labels
                        freq[lab] = freq.get(lab, 0) + 1
                # Extract labels that occur two or more times, saving each label only once.
                repeated = [lab for lab, count in freq.items() if count >= 2]
            else:
                # Similarity matching
                # We'll use a more complex structure to track similar labels
                freq = {}  # Maps canonical labels to counts
                label_groups = {}  # Maps canonical labels to sets of similar labels
                child_labels = {}  # Maps child IDs to their labels for reference
                
                for child in node.kids:
                    lab = child.label
                    if not lab:  # Skip empty labels
                        continue
                    
                    child_labels[child.id] = lab
                    
                    # Check if this label is similar to any existing canonical label
                    similar_found = False
                    for canonical_label in list(freq.keys()):
                        if labels_are_similar(lab, canonical_label, 
                                             method=similarity_method, 
                                             threshold=similarity_threshold):
                            freq[canonical_label] += 1
                            label_groups.setdefault(canonical_label, set()).add(lab)
                            similar_found = True
                            break
                    
                    # If no similar label found, add this as a new canonical label
                    if not similar_found:
                        freq[lab] = 1
                        label_groups[lab] = {lab}
                
                # Extract canonical labels that occur two or more times
                repeated = []
                for lab, count in freq.items():
                    if count >= 2:
                        # Include the canonical label and all its similar variants
                        repeated.append({
                            "canonical": lab,
                            "variants": list(label_groups[lab]),
                            "normalized": normalize_label(lab)
                        })
            
            if repeated:
                results.append({
                    "parent": (node.id, node.label),
                    "kids": [(kid.id, kid.label) for kid in node.kids],
                    "repeated_labels": repeated
                })
        # Recurse on each child.
        for child in node.kids:
            helper(child)
    
    helper(root)
    return results


def get_deepest_level_threshold(node):
    """
    Recursively computes the deepest level among nodes that are considered (node.level != -1)
    in the flexible tree and returns the nodes at that level.
    
    If a node is not considered (level == -1), that branch is ignored.
    
    Returns:
        tuple: (deepest_level, list of nodes at deepest level)
    """
    if node is None or node.level == -1:
        return -1, []  # -1 indicates that the branch is not considered
    
    deepest = node.level
    deepest_nodes = []
    
    for child in node.kids:
        child_level, child_nodes = get_deepest_level_threshold(child)
        if child_level > deepest:
            deepest = child_level
            deepest_nodes = child_nodes
        elif child_level == deepest:
            deepest_nodes.extend(child_nodes)
    
    # If this node is at the deepest level seen so far, include it
    if node.level == deepest:
        deepest_nodes.append(node)
        
    return deepest, deepest_nodes


def get_ancestry_chain_nary(root, target_id):
    """
    Returns a list of nodes from the root to the target node (inclusive).
    Works for a flexible tree (nodes have a kids list).
    If target not found, returns an empty list.
    """
    chain = []
    def helper(node):
        if node is None:
            return False
        chain.append(node)
        if node.id == target_id:
            return True
        for child in node.kids:
            if helper(child):
                return True
        chain.pop()
        return False
    return chain if helper(root) else []


def print_path_to_node(root, target_id):
    """
    Prints the path from the root to the node with the given target_id in the format:
      (id:label) -> (id:label) -> ... -> (id:label)
    
    If the node is not found, prints an appropriate message.
    """
    chain = get_ancestry_chain_nary(root, target_id)
    if not chain:
        print(f"Node with id {target_id} not found.")
        return
    # Build the path string by joining the representations of each node.
    path_str = " -> \n".join(f"({node.id}:{node.label if node.label else 'None'})" for node in chain)
    print(path_str)


def build_ascii_tree_nary(node, max_level, current_level=0, prefix="", is_last=True):
    """
    Recursively builds an ASCII tree representation of the subtree rooted at 'node'
    up to 'max_level' (where current_level=0 is the root of this subtree).
    
    This version works for flexible trees (nodes have a kids list).
    
    Parameters:
      - node: the current tree node.
      - max_level: maximum depth to print (0 means only the node itself).
      - current_level: the current depth (used in recursion).
      - prefix: string prefix for the current line.
      - is_last: boolean indicating if the node is the last child of its parent.
    
    Returns a string with the ASCII representation.
    """
    if node is None or current_level > max_level:
        return ""
    
    if current_level == 0:
        result = f"{node.label if node.label else 'None'} (id: {node.id})\n"
    else:
        connector = "└── " if is_last else "├── "
        result = prefix + connector + f"{node.label if node.label else 'None'} (id: {node.id})\n"
    
    new_prefix = prefix + ("    " if is_last else "│   ") if current_level > 0 else ""
    # Use node.kids directly.
    for i, child in enumerate(node.kids):
        child_is_last = (i == len(node.kids) - 1)
        result += build_ascii_tree_nary(child, max_level, current_level + 1, new_prefix, child_is_last)
    return result


def find_node_and_parent_nary(root, target_id, parent=None):
    """
    Recursively searches for the node with target_id in a flexible tree.
    Returns (node, parent). If the target is the root, returns (root, None).
    """
    if root is None:
        return None, None
    if root.id == target_id:
        return root, parent
    for child in root.kids:
        found, par = find_node_and_parent_nary(child, target_id, root)
        if found is not None:
            return found, par
    return None, None


def print_upward_context_ascii_nary(root, target_id, levels_up=1):
    """
    Prints the upward (ancestral) context as an ASCII tree.
    
    1. Gets the ancestry chain from the root to the target.
    2. Selects an ancestor node that is `levels_up` above the target (if possible).
    3. Uses build_ascii_tree to print that ancestor's subtree with a depth 
       equal to the distance from that ancestor to the target.
    """
    chain = get_ancestry_chain_nary(root, target_id)
    if not chain:
        print(f"Node with id {target_id} not found.")
        return
    # For levels_up=1 and chain=[Grandparent, Parent, Target], choose the Parent.
    index = max(0, len(chain) - levels_up - 1)
    upward_root = chain[index]
    depth = len(chain) - index - 1  # distance from upward_root to target
    print("=== UPWARD CONTEXT ===")
    ascii_tree = build_ascii_tree_nary(upward_root, max_level=depth)
    print(ascii_tree)


def print_downward_context_ascii_nary(root, target_id, levels_down=2):
    """
    Prints the downward (descendant) context as an ASCII tree.
    
    Finds the target node in the flexible tree and prints its subtree
    with a depth defined by levels_down.
    """
    target, _ = find_node_and_parent_nary(root, target_id)
    if target is None:
        print(f"Node with id {target_id} not found.")
        return
    print("=== DOWNWARD CONTEXT ===")
    ascii_tree = build_ascii_tree_nary(target, max_level=levels_down)
    print(ascii_tree)


def print_context_flexible_nary(root, target_id, levels_up=1, levels_down=2):
    """
    Prints a flexible visualization of the target node's context:
      - Upward: an ASCII tree for an ancestor (levels_up above target) down to the target.
      - Downward: an ASCII tree for the target's descendants (up to levels_down).
    
    Both contexts are printed separately.
    """
    print_upward_context_ascii_nary(root, target_id, levels_up)
    print("=== TARGET NODE ===")
    target, _ = find_node_and_parent_nary(root, target_id)
    if target:
        # Print only the target node (without its children).
        print(build_ascii_tree_nary(target, max_level=0))
    print()
    print_downward_context_ascii_nary(root, target_id, levels_down)


def classify_within_branch_repetition(trimmed):
    """
    Given a trimmed branch (list of (node.id, label) tuples) with no trailing empty labels,
    classify its repetition pattern for the meaningful labels.
    
    Returns:
      - "intervening" if there is any repeated label with a gap.
      - "consecutive" if there is at least one repeated label and all repetitions are adjacent.
      - "none" if there are no repeated labels.
    """
    # Extract only meaningful labels.
    labels = [lbl for (_, lbl) in trimmed if lbl]
    
    # Build a mapping from each label to the list of indices where it appears.
    index_map = {}
    for i, lbl in enumerate(labels):
        index_map.setdefault(lbl, []).append(i)
    
    has_repetition = False
    for lbl, indices in index_map.items():
        if len(indices) > 1:
            has_repetition = True  # There is repetition for this label.
            # Check if there is any gap between successive occurrences.
            for j in range(1, len(indices)):
                if indices[j] - indices[j-1] > 1:
                    return "intervening"
    return "consecutive" if has_repetition else "none"


def classify_result(result):
    """
    Given a result from results_leaf (which contains the full path),
    trim the path and classify the repetition pattern.
    """
    full_path = result.get("path", [])
    trimmed = trim_trailing_empty(full_path)
    return classify_within_branch_repetition(trimmed)


def convert_to_d3_format(node, parent_name=None):
    """
    Recursively converts a FlexibleTreeNode (with attributes: id, label, data_points, level, dist, kids)
    into a dictionary in the D3.js tree format:
      - "parent": parent's name (or "null" if no parent),
      - "name": a string combining the node's label and id,
      - "edge_name": a string (here we use node.data_points if available),
      - "children": list of converted children.
    """
    # Set the parent's name to "null" if None.
    parent_str = parent_name if parent_name is not None else "null"
    
    # Build the node's "name" as "label (id)" (use "None" if label is empty)
    name_str = f"{node.id}: {node.label if node.label else 'None'}"
    
    # For edge_name, use the string form of data_points if available, else "null"
    edge_str = ''
    
    sampled_titles = str(node.sampled_titles) if hasattr(node, 'sampled_titles') else "null"
    
    # Recursively convert each child, passing the current node's name as the parent.
    children_list = [convert_to_d3_format(child, name_str) for child in node.kids]

    
    return {
        "parent": parent_str,
        "name": name_str,
        "edge_name": edge_str,
        "sampled_titles": sampled_titles,
        "children": children_list
    }

def convert_flexible_tree_to_d3(root):
    """
    Converts the entire FlexibleTreeNode tree (root) to a D3.js-compatible dictionary.
    """
    return convert_to_d3_format(root)

######################
# n-ary tree pruning
######################
def labels_are_similar(label1, label2, method='combined', threshold=0.85):
    """
    Compare two labels using multiple similarity methods.
    
    Args:
        label1, label2: The labels to compare
        method: 'combined', 'spacy', or 'fuzzy'
        threshold: Similarity threshold (0.0 to 1.0)
    """
    if label1 == '' or label2 == '':
        return False
        
    # Normalize both labels
    norm1 = normalize_label(label1)
    norm2 = normalize_label(label2)
    
    if not norm1 or not norm2:
        return False

    if method == 'combined':
        # Try multiple similarity metrics and use the highest score
        
        # 1. Word-based Jaccard similarity
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        if words1.issubset(words2) or words2.issubset(words1):
            return True
            
        # 2. Fuzzy token set ratio (handles word reordering and partial matches)
        if fuzz.token_set_ratio(norm1, norm2) >= threshold * 100:
            return True
            
        # 3. Fuzzy partial ratio (handles substrings)
        if fuzz.partial_ratio(norm1, norm2) >= threshold * 100:
            return True
            
        return False
        
    elif method == 'spacy':
        # Use spaCy's word vectors for semantic similarity
        nlp = spacy.load('en_core_web_md')
        doc1 = nlp(norm1)
        doc2 = nlp(norm2)
        return doc1.similarity(doc2) >= threshold
        
    elif method == 'fuzzy':
        # Use token set ratio from thefuzz
        return fuzz.token_set_ratio(norm1, norm2) >= threshold * 100
        
    return False

def normalize_label(label):
    """Enhanced normalization with common variations."""
    if not label:
        return ''
    
    # Convert to lowercase
    label = label.lower()
    
    # Remove special characters and extra whitespace
    label = re.sub(r'[^\w\s]', ' ', label)
    
    # Common word replacements
    replacements = {
        'educational': 'education',
        'administrative': 'admin',
        'administration': 'admin',
        'management': 'manager',
        'engineering': 'engineer',
        'development': 'developer',
        'technical': 'tech',
        'technology': 'tech',
        'professional': '',
        'specialist': '',
        'associate': '',
    }
    
    # Split into words
    words = label.split()
    
    # Remove common job title suffixes
    suffix_words = {
        'roles', 'role', 'positions', 'position', 'jobs', 'job',
        'staff', 'workers', 'worker', 'professionals', 'professional',
        'team', 'department', 'dept', 'group'
    }
    
    # Apply replacements and remove suffixes
    normalized_words = []
    for word in words:
        word = word.rstrip('s')  # Remove trailing 's'
        if word not in suffix_words:
            word = replacements.get(word, word)
            if word:  # Only add if not empty
                normalized_words.append(word)
    
    return ' '.join(normalized_words).strip()


def prune_consecutive_repetition(node, method='exact', threshold=0.85):
    """
    Recursively prune nodes whose label is similar to their parent's label.
    
    Args:
        node: The tree node to process
        method: Similarity method to use:
            - 'exact': Strict string equality
            - 'combined': Multiple similarity metrics (fuzzy + word-based)
            - 'spacy': Semantic similarity using word vectors
            - 'fuzzy': Fuzzy string matching
        threshold: Similarity threshold (0.0 to 1.0) for non-exact methods
    """
    if node is None:
        return None
    
    pruned_children = []
    for child in node.kids:
        pruned_child = prune_consecutive_repetition(child, method, threshold)
        
        # Skip empty labels
        if not pruned_child or pruned_child.label == '':
            if pruned_child:
                pruned_children.append(pruned_child)
            continue
            
        # Compare labels based on selected method
        labels_match = False
        if method == 'exact':
            labels_match = pruned_child.label == node.label
        else:
            labels_match = labels_are_similar(pruned_child.label, node.label, method, threshold)
        
        # If labels match, merge child's kids into parent
        if labels_match:
            pruned_children.extend(pruned_child.kids)
        else:
            pruned_children.append(pruned_child)
            
    node.kids = pruned_children
    return node


def prune_consecutive_repetition_until_stable(node, method='exact', threshold=0.85):
    """Run pruning passes until the tree structure stabilizes."""
    
    # Function to count nodes in tree for comparison
    def count_nodes(n):
        if not n:
            return 0
        return 1 + sum(count_nodes(kid) for kid in n.kids)
    
    prev_count = -1
    current_count = count_nodes(node)
    
    # Continue pruning until no more nodes are removed
    while prev_count != current_count:
        prev_count = current_count
        node = prune_consecutive_repetition(node, method, threshold)
        current_count = count_nodes(node)
        
    return node


def merge_repeated_siblings(node, method='exact', threshold=0.85):
    """
    Recursively merge siblings with similar labels in a flexible tree.
    
    For each node, if two or more children have similar labels,
    merge them into a single child:
      - The first occurrence is used as the base.
      - Append the kids of later siblings (with similar labels)
        to the base's kids list.
    
    Args:
        node: The tree node to process
        method: Similarity method to use:
            - 'exact': Strict string equality
            - 'combined': Multiple similarity metrics (fuzzy + word-based)
            - 'spacy': Semantic similarity using word vectors
            - 'fuzzy': Fuzzy string matching
        threshold: Similarity threshold (0.0 to 1.0) for non-exact methods
    """
    if node is None:
        return None

    # First, merge in the subtrees of all children
    for child in node.kids:
        merge_repeated_siblings(child, method, threshold)
    
    # Now, merge similar siblings at the current node
    merged = {}  # Maps label -> base child node
    new_kids = []
    
    for child in node.kids:
        if child.label == '':
            # Do not merge nodes with empty label
            new_kids.append(child)
            continue
            
        # Check if this label is similar to any existing merged label
        found_match = False
        for existing_label in merged:
            if method == 'exact':
                labels_match = child.label == existing_label
            else:
                labels_match = labels_are_similar(child.label, existing_label, method, threshold)
                
            if labels_match:
                # Merge this child's kids into the existing base node
                merged[existing_label].kids.extend(child.kids)
                found_match = True
                break
                
        if not found_match:
            # No similar label found, add as new base node
            merged[child.label] = child
            new_kids.append(child)
    
    node.kids = new_kids
    return node


def merge_repeated_siblings_until_stable(node, method='exact', threshold=0.85):
    """
    Run sibling merging passes until the tree structure stabilizes.
    
    Args:
        node: The tree node to process
        method: Similarity method to use
        threshold: Similarity threshold for non-exact methods
    
    Returns:
        The processed tree with all similar siblings merged
    """
    # Function to get a signature of the tree structure for comparison
    def tree_signature(n):
        if not n:
            return "None"
        
        # Create a signature based on node label and number of children
        sig = f"({n.label}:{len(n.kids)})"
        
        # Add signatures of all children, sorted for consistency
        child_sigs = [tree_signature(kid) for kid in n.kids]
        child_sigs.sort()
        
        return sig + "-".join(child_sigs)
    
    prev_signature = ""
    current_signature = tree_signature(node)
    
    # Continue merging until no more changes are detected
    passes = 0
    while prev_signature != current_signature and passes < 10:  # Limit to 10 passes to prevent infinite loops
        passes += 1
        prev_signature = current_signature
        
        # Perform one pass of merging
        node = merge_repeated_siblings(node, method, threshold)
        
        # Check if the tree structure changed
        current_signature = tree_signature(node)
        
    return node

