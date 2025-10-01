#!/usr/bin/env python3
"""
Taxonomy Evaluation Script

This script processes taxonomy data, reformats it for LLM input, generates prompts,
and calls different LLMs to annotate job postings.
"""

import csv
import json
import pickle
import pandas as pd
import os
import random
from io import StringIO
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from tqdm import tqdm # Added for progress bar
from itertools import combinations
from collections import Counter

# Attempt to import OpenAI; fail gracefully if not installed
try:
    from openai import OpenAI
except ImportError:
    print("OpenAI library not found. Please install it: pip install openai")
    OpenAI = None

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"  # Assuming data is in a 'data' subdirectory
TMP_DIR = Path("../tmp")
CLUSTER_DIR = TMP_DIR / "xgbt_clustering"

# Ensure output directory exists
# OUTPUT_DIR.mkdir(exist_ok=True)

# --- Initialize API Clients (if library is available) ---
openai_client: Optional[OpenAI] = None
deepseek_client: Optional[OpenAI] = None

if OpenAI:
    if os.environ.get("OPENAI_API_KEY"):
        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    else:
        print("Warning: OPENAI_API_KEY environment variable not set. OpenAI calls will fail.")

    if os.environ.get("DEEPSEEK_API_KEY"):
        try:
            deepseek_client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1" # Corrected base_url
            )
        except Exception as e:
            print(f"Warning: Could not initialize DeepSeek client. Error: {e}")
    else:
        print("Warning: DEEPSEEK_API_KEY environment variable not set. DeepSeek calls will fail.")

# --- Helper Functions ---
def get_isco_level_from_code(isco_code_str: str) -> Optional[int]:
    """Calculates ISCO level (0-3 for ISCO groups) based on code length."""
    length = len(isco_code_str)
    if length == 1: return 0
    if length == 2: return 1
    if length == 3: return 2
    if length == 4: return 3
    return None

# --- Taxonomy Reformatting Functions ---

def reformat_esco_under_isco_tree(occupations_csv_str, isco_groups_csv_str, min_level=3, desc_len=40, max_level=None):
    """
    Reformats ESCO/ISCO taxonomy, including only nodes between the specified minimum and maximum levels.
    
    Args:
        occupations_csv_str: CSV string containing ESCO occupations
        isco_groups_csv_str: CSV string containing ISCO groups
        min_level: Minimum ISCO level to include (0-3 for ISCO groups, default=3)
        max_level: Maximum ISCO level to include (default=None, meaning no upper limit)
    
    Returns:
        tuple: (formatted taxonomy string, mapping of LLM IDs to original codes, all_nodes dict, node count)
    """
    all_nodes = {}  # Stores all ISCO and ESCO nodes. Key: unique_id (ISCO code or ESCO URI)
    node_count = 0  # Counter for total nodes kept
    level_counts = {}  # Counter for nodes per level

    # --- 1. Parse ISCO Groups (parse all, filter later) ---
    reader_isco = csv.DictReader(StringIO(isco_groups_csv_str))
    for row in reader_isco:
        code = row['code'].strip()
        level = get_isco_level_from_code(code)
        if level is None:
            print(f"Warning: Skipping ISCO group with invalid code '{code}'")
            continue
        
        all_nodes[code] = {
            'id_for_llm': '',  # Will be populated during tree traversal
            'title': row['preferredLabel'].strip(),
            'description': row['description'].strip().replace('\n', ' ').replace('  ', ' '),
            'level': level, # This is ISCO's internal level, not final tree depth yet
            'children_keys': [],
            'original_code': code,
            'type': 'ISCO'
        }
        
        # Count all nodes by level for reporting
        level_counts[level] = level_counts.get(level, 0) + 1

    # --- 2. Build ISCO Hierarchy (Parent-Child Links for ISCO Groups) ---
    # Sort by code length to process parents before children for ISCO groups
    sorted_isco_codes = sorted([k for k, v in all_nodes.items() if v['type'] == 'ISCO'], key=lambda x: (len(x), x))
    for code in sorted_isco_codes:
        if len(code) > 1:  # Not a top-level group
            parent_code = code[:-1]
            if parent_code in all_nodes and all_nodes[parent_code]['type'] == 'ISCO':
                if code not in all_nodes[parent_code]['children_keys']:
                    all_nodes[parent_code]['children_keys'].append(code)

    # --- 3. Parse ESCO Occupations and prepare for tree insertion ---
    esco_occupations_temp = {} # Temp store by full ESCO code
    reader_esco = csv.DictReader(StringIO(occupations_csv_str))
    for row in reader_esco:
        full_esco_code = row['code'].strip()
        if not full_esco_code:
            continue
        
        # Use conceptUri as a unique internal key for ESCO occupations to avoid code collisions
        esco_key = row['conceptUri']
        
        # For ESCO codes, check if the base ISCO part meets the minimum level
        parts = full_esco_code.split('.')
        base_isco_part = parts[0]
        base_isco_level = get_isco_level_from_code(base_isco_part)
        
        # Calculate the ESCO level based on the base ISCO level and the number of dot-separated parts
        esco_level = base_isco_level + len(parts) - 1 if base_isco_level is not None else None
        
        if esco_level is not None:
            esco_occupations_temp[full_esco_code] = {
                'esco_key': esco_key, # Store the URI to use as the actual key in all_nodes
                'title': row['preferredLabel'].strip(),
                'description': (row['definition'] if row.get('definition', '').strip() else row['description']).strip().replace('\n', ' ').replace('  ', ' '),
                'children_keys': [],
                'original_code': full_esco_code,
                'type': 'ESCO',
                'level': esco_level,
                'id_for_llm': ''  # Initialize with empty string
            }
            
            # Count all nodes by level for reporting
            level_counts[esco_level] = level_counts.get(esco_level, 0) + 1

    # --- 4. Integrate ESCO Occupations into the `all_nodes` tree structure ---
    # First, add all ESCO occupations to all_nodes using their unique URI
    for full_esco_code, data in esco_occupations_temp.items():
        all_nodes[data['esco_key']] = data # URI is the key
        
    # Then, establish parent-child relationships for ESCO nodes
    sorted_esco_codes_for_linking = sorted(esco_occupations_temp.keys(), key=lambda x: (len(x.split('.')), x))

    for full_esco_code in sorted_esco_codes_for_linking:
        current_esco_data = esco_occupations_temp[full_esco_code]
        current_esco_key = current_esco_data['esco_key']
        
        parts = full_esco_code.split('.')
        
        if len(parts) == 1: # e.g. "2654" (if an ESCO occ has such a code, rare)
            isco_parent_code = parts[0][:4] # Ensure it's max 4 digits
            if isco_parent_code in all_nodes and all_nodes[isco_parent_code]['type'] == 'ISCO':
                if current_esco_key not in all_nodes[isco_parent_code]['children_keys']:
                    all_nodes[isco_parent_code]['children_keys'].append(current_esco_key)
        else: # ESCO code has dots, e.g., 2654.1 or 2654.1.7
            parent_esco_code_candidate = ".".join(parts[:-1])
            
            if parent_esco_code_candidate in esco_occupations_temp: # Parent is another ESCO occ
                parent_esco_key = esco_occupations_temp[parent_esco_code_candidate]['esco_key']
                if current_esco_key not in all_nodes[parent_esco_key]['children_keys']:
                    all_nodes[parent_esco_key]['children_keys'].append(current_esco_key)
            elif parts[0] in all_nodes and all_nodes[parts[0]]['type'] == 'ISCO': # Parent is the 4-digit ISCO code
                isco_parent_key = parts[0]
                if current_esco_key not in all_nodes[isco_parent_key]['children_keys']:
                    all_nodes[isco_parent_key]['children_keys'].append(current_esco_key)

    # --- 5. Sort children for consistent output ---
    for node_key in all_nodes:
        # Sort by original_code; ISCO codes are numeric-like, ESCO codes are dot-separated
        all_nodes[node_key]['children_keys'].sort(key=lambda k: all_nodes[k]['original_code'])
    
    # --- 6. Find top-level nodes based on min_level ---
    # Now that we have the complete hierarchy, identify nodes at min_level to use as top-level nodes
    top_level_keys = []
    for node_key, node_data in all_nodes.items():
        if node_data['level'] == min_level:
            top_level_keys.append(node_key)
    
    # Sort top-level keys for consistent output
    top_level_keys.sort(key=lambda k: all_nodes[k]['original_code'])
    
    print(f"Available levels: {sorted(level_counts.keys())}")
    print(f"Found {len(top_level_keys)} top-level nodes at level {min_level}")
    
    if not top_level_keys:
        print(f"Warning: No nodes found at level {min_level}. Check your min_level parameter.")
        return "", {}, all_nodes, 0
    
    # --- 7. Filter nodes by level and build the LLM string ---
    output_lines = []
    visited_nodes = set()  # Track which nodes have been visited
    final_level_counts = {}  # Counter for nodes per final level in the tree
    nodes_kept_count = 0  # Counter for nodes kept in the final tree
    
    def generate_llm_string_recursive(node_key, current_depth_level, path_str_so_far, indent_level):
        nonlocal nodes_kept_count  # Use nonlocal to modify the counter
        
        node_data = all_nodes[node_key]
        node_level = node_data['level']
        
        # Skip nodes outside the specified level range
        if node_level < min_level or (max_level is not None and node_level > max_level):
            return
        
        node_data['final_level'] = current_depth_level # Assign true depth in the combined tree
        node_data['id_for_llm'] = f"L{current_depth_level}.{path_str_so_far}"
        visited_nodes.add(node_key)  # Mark this node as visited
        nodes_kept_count += 1  # Count this node as kept
        
        # Count nodes by final tree level
        final_level_counts[current_depth_level] = final_level_counts.get(current_depth_level, 0) + 1
        
        # Clean up description
        desc = node_data['description'].replace('\n', ' ').replace('\r', ' ').replace('  ', ' ').strip()
        desc = ' '.join(desc.split(' ')[:desc_len])

        output_lines.append(
            f"{'  ' * indent_level}{node_data['id_for_llm']}: Title: {node_data['title']} - Description: {desc} ..."
        )

        # Process children that are within the level range
        valid_children = []
        for child_key in node_data['children_keys']:
            child_level = all_nodes[child_key]['level']
            if child_level >= min_level and (max_level is None or child_level <= max_level):
                valid_children.append(child_key)
        
        for i, child_key in enumerate(valid_children):
            new_path_str = f"{path_str_so_far}.{i+1}" if path_str_so_far else str(i+1)
            generate_llm_string_recursive(child_key, current_depth_level + 1, new_path_str, indent_level + 1)

    # Start recursion for top-level nodes
    for i, root_key in enumerate(top_level_keys):
        generate_llm_string_recursive(root_key, 0, str(i + 1), 0) # L0 for the top level groups
    
    # Check for unvisited nodes and assign them placeholder IDs
    for node_key, node_data in all_nodes.items():
        if node_key not in visited_nodes:
            node_data['id_for_llm'] = f"UNVISITED_{node_data['original_code']}"
    
    # save id for llm to code mapping (only for visited nodes)
    id_for_llm_to_code_mapping = {node_data['id_for_llm']: node_data['original_code'] 
                                 for node_key, node_data in all_nodes.items() 
                                 if node_key in visited_nodes}

    # Print statistics
    print(f"Total nodes kept: {nodes_kept_count}")
    print(f"Nodes in taxonomy tree: {len(visited_nodes)}")
    
    # Print nodes per original level
    print("\nNodes per original level (before tree construction):")
    for level in sorted(level_counts.keys()):
        print(f"  Level {level}: {level_counts[level]} nodes")
    
    # Print nodes per final level in the tree
    print("\nNodes per final level in the tree:")
    for level in sorted(final_level_counts.keys()):
        print(f"  Level {level}: {final_level_counts[level]} nodes")
    
    return "\n".join(output_lines), id_for_llm_to_code_mapping, all_nodes, nodes_kept_count

# re-format the taxonomy tree for LLM input
def reformat_tree_taxonomy_for_llm(taxonomy_data, title2desc, desc_len=50, add_examples=False, permute_branches=False):
    """
    Reformats the hierarchical taxonomy data into a string representation
    suitable for LLM prompting. Each node gets a unique path-based ID.

    Args:
        taxonomy_data (dict): The input taxonomy as loaded from the JSON.
        title2desc (dict): A mapping from title to description.
        desc_len (int): The maximum number of words to include in the description.
        add_examples (bool): Whether to include examples in the output.
        permute_branches (bool): If True, shuffle the order of branches at each level.

    Returns:
        tuple: A tuple containing:
            - str: A string representation of the taxonomy.
            - dict: A dictionary mapping the generated string ID to the original title.
    """
    if not taxonomy_data:
        return "Taxonomy is empty.", {}

    # Use a (title, level) tuple as the key to handle duplicate titles across levels
    all_items_by_key = {}
    id_to_title_map = {}

    # 1. First, populate `all_items_by_key` with all nodes from the taxonomy data.
    # We use a (title, level) tuple for the key to ensure uniqueness.
    
    # Process leaf items (Level 0)
    if "0" in taxonomy_data and taxonomy_data["0"]['Leaf Items']:
        for item in taxonomy_data["0"]['Leaf Items']:
            key = (item["title"], 0)
            if key not in all_items_by_key:
                all_items_by_key[key] = {
                    "description": item.get("description", ""),
                    "examples": item.get("examples", []),
                    "examples_titles": item.get("examples_titles", []),
                    "level": 0,
                    "kids": []  # Kids will be populated later
                }

    # Process non-leaf items (Levels 1 and up)
    levels = sorted([int(k) for k in taxonomy_data.keys() if k != "0"])
    for level_num in levels:
        level_key = str(level_num)
        if level_key in taxonomy_data:
            for item_data in taxonomy_data[level_key]:
                key = (item_data["title"], level_num)
                if key not in all_items_by_key:
                    all_items_by_key[key] = {
                        "description": title2desc.get(item_data["title"], "N/A"),
                        "level": level_num,
                        "kids": item_data.get("kids", [])  # Store kid titles temporarily
                    }

    # 2. Now, create a map from title to a list of keys to find nodes efficiently.
    title_to_keys = {}
    for key in all_items_by_key:
        title, level = key
        if title not in title_to_keys:
            title_to_keys[title] = []
        title_to_keys[title].append(key)

    # 3. Build the final hierarchy by resolving kid titles to their unique keys.
    for parent_key, parent_node in all_items_by_key.items():
        parent_level = parent_node["level"]
        kid_titles = parent_node["kids"]  # These are still strings
        resolved_kid_keys = []
        for kid_title in kid_titles:
            # A kid of a level N node should be at level N-1.
            # If not found, it could be a leaf node (level 0).
            expected_kid_level = parent_level - 1
            kid_key_candidate = (kid_title, expected_kid_level)
            
            if kid_key_candidate in all_items_by_key:
                resolved_kid_keys.append(kid_key_candidate)
            else:
                # Check if it's a leaf node. This handles cases where L2+ nodes have L0 children.
                leaf_key_candidate = (kid_title, 0)
                if leaf_key_candidate in all_items_by_key:
                    resolved_kid_keys.append(leaf_key_candidate)
                else:
                    # If the kid is not defined, we can log a warning.
                    print(f"Warning: Child '{kid_title}' of parent '{parent_key[0]}' not found at level {expected_kid_level} or as a leaf.")
        
        parent_node["kids"] = resolved_kid_keys  # Update kids from titles to unique keys

    # --- The rest of the function remains similar but uses the new key system ---
    output_lines = []
    
    # Identify top-level nodes for starting the recursion.
    if not levels: # Only leaf items
        top_level_keys = [key for key in all_items_by_key.keys() if key[1] == 0]
    else:
        highest_level_num = max(levels)
        top_level_keys = [key for key in all_items_by_key.keys() if key[1] == highest_level_num]

    if permute_branches:
        print("Permuting top-level branches of the tree.")
        random.shuffle(top_level_keys)
    else:
        # Sort by title for consistency
        top_level_keys.sort(key=lambda k: k[0])

    # --- Identify and warn about duplicate titles across levels ---
    titles_at_levels = {}
    for title, level in all_items_by_key.keys():
        if title not in titles_at_levels:
            titles_at_levels[title] = []
        titles_at_levels[title].append(level)
    
    duplicate_titles = {title: levels for title, levels in titles_at_levels.items() if len(levels) > 1}
    
    if duplicate_titles:
        print("Warning: The following titles are duplicated across multiple levels:")
        for title, levels in sorted(duplicate_titles.items()):
            print(f"- Title: '{title}' found at levels: {sorted(levels)}")

    # Recursive function to build the string
    def build_string_recursive(node_key, current_path_id, indent_level, desc_len=50):
        node_data = all_items_by_key.get(node_key)
        if not node_data:
            print(f"Warning: Node with key '{node_key}' mentioned but not found.")
            return

        level_prefix = f"L{node_data['level']}."
        full_id = level_prefix + current_path_id
        id_to_title_map[full_id] = node_key[0] # Map the ID to the title string

        if desc_len < 0:
            desc = node_data['description']
            desc_suffix = ""
        else:
            desc = ' '.join(node_data['description'].split(' ')[:desc_len])
            desc_suffix = " ..."

        examples_titles_str = ""
        examples_str = ""
        if add_examples:
            examples_titles = node_data.get("examples_titles", [])
            examples_titles_str = ", ".join([f"{example_title}" for example_title in examples_titles])

            examples = node_data.get("examples", [])
            examples_str = "\n".join([f"{example}" for example in examples])
            output_lines.append(
                f"{'  ' * indent_level}{full_id}: Title: {node_key[0]} - Description: {desc}{desc_suffix}" + \
                (f"==========\nExample Titles: {examples_titles_str}\n{examples_str}" if examples_str else "")
            )
        else:
            output_lines.append(
                f"{'  ' * indent_level}{full_id}: Title: {node_key[0]} - Description: {desc}{desc_suffix}"
            )

        # Sort or shuffle kids for consistent output. Kids are now keys (title, level).
        kid_keys = list(node_data.get("kids", []))
        if permute_branches:
            random.shuffle(kid_keys)
        else:
            kid_keys.sort(key=lambda k: k[0])

        for i, kid_key in enumerate(kid_keys):
            new_path_id_suffix = f"{i+1}"
            new_path_segment = f"{current_path_id}.{new_path_id_suffix}" if current_path_id else new_path_id_suffix
            build_string_recursive(kid_key, new_path_segment, indent_level + 1, desc_len=desc_len)

    # Start recursion for top-level nodes
    for i, key in enumerate(top_level_keys):
        build_string_recursive(key, str(i + 1), 0, desc_len=desc_len)
        
        
    print(f"Total unique nodes processed: {len(all_items_by_key)}")

    return "\n".join(output_lines), id_to_title_map


def reformat_flat_taxonomy_for_llm(flat_taxonomy_items: List[Dict[str, str]], desc_len=20) -> str:
    """
    Reformats a flat list of taxonomy items for LLM input.
    Each item in the list should be a dictionary with 'title' and 'description'.
    """
    output_lines: List[str] = []
    id_to_title_map = {}
    if not flat_taxonomy_items:
        return "Taxonomy list is empty."

    for i, item in enumerate(flat_taxonomy_items):
        title = item.get("name", "N/A")
        description = item.get("description", "N/A")
        desc_short = ' '.join(description.split(' ')[:desc_len]) + \
                     ('...' if len(description.split(' ')) > desc_len else '')
        output_lines.append(f"L0.{i+1}: Title: {title} - Description: {desc_short} ...")
        id_to_title_map[f"L0.{i+1}"] = title
    return "\n".join(output_lines), id_to_title_map

# --- Prompt Generation Functions ---

def get_tree_annotation_prompt(job_posting_str, taxonomy_str):
    annotation_prompt = f"""You are an expert job classification system. Your task is to analyze the provided job posting and assign it to the most appropriate occupation(s) from the given hierarchical taxonomy.

Carefully review the entire job posting and the provided taxonomy.

**Taxonomy Structure:**
Each occupation in the taxonomy has a unique ID (e.g., L2.1, L1.1.1), a Title, and a Description.
In addition, some occupations may be followed by a list of example job titles and a full example job posting for context.

**Important Guidance on Using Examples:**
The provided examples are for illustrative purposes. They can sometimes be noisy or not a perfect representation of the occupation. **You must base your final decision on the official 'Title' and 'Description' of the occupation.** Use the examples only as a secondary reference to aid your understanding, not as the primary basis for your classification.

**Annotation Rules:**

1.  **Primary Goal:** Identify the single most appropriate occupation from the taxonomy that best describes the core responsibilities and nature of the job posting.
2.  **Granularity:**
    *   If the job posting provides enough specific detail, assign it to the most granular leaf occupation (L0 level) that accurately matches.
    *   If the job posting is too vague for a specific L0 leaf occupation but clearly aligns with a broader parent occupation (L1, L2, etc.), assign it to that parent occupation. Choose the most specific parent occupation that is still a good fit.
3.  **Multiple Occupations (Use VERY Sparingly):**
    *   Ideally, assign only one occupation.
    *   However, if the job posting genuinely and substantially describes responsibilities that clearly map to *two or more distinct occupations* within the taxonomy, and a single occupation would be insufficient, you may list multiple occupations.
    *   If listing multiple, ensure each choice is a strong fit. Do not list a parent and its child simultaneously.
4.  **"Other" Category:** If, after careful consideration, no occupation in the provided taxonomy (at any level) is a suitable fit for the job posting, assign it as "Other".

**Output Format:**

Provide your response as a JSON object with the following structure:

```json
{{{{
"assigned_occupations": [
    // This array will contain occupation objects if a match is found.
    // Example: {{"id": "L0.X.Y.Z", "title": "Title of the assigned occupation"}}
    // This array will be EMPTY if no match is found (and is_other is true).
],
"is_other": true/false // true if no suitable occupation is found, false otherwise.
}}}}

Example of assigned_occupations array entries:
Single leaf assignment: {{{{
"assigned_occupations": [{{"id": "L0.1.1.1", "title": "Care Coordinator"}}],
"is_other": false
}}}}
Single parent assignment: {{{{
"assigned_occupations": [{{"id": "L1.1.1", "title": "Healthcare Direct Care and Social Services"}}],
"is_other": false
}}}}
Multiple assignments (extremely rare): {{{{
"assigned_occupations": [
    {{"id": "L0.X.Y.Z", "title": "Specific Role A"}},
    {{"id": "L0.A.B.C", "title": "Specific Role B"}}
],
"is_other": false
}}}}
Other: {{{{
"assigned_occupations": [], // Empty array
"is_other": true
}}}}

---START OF TAXONOMY---
{taxonomy_str}
---END OF TAXONOMY---

---START OF JOB POSTING---
{job_posting_str}
---END OF JOB POSTING---

Your response MUST be a single, valid JSON object and nothing else. Do NOT include any explanatory text or markdown formatting before or after the JSON object.
Now, please provide the JSON output for the job posting above.
"""
    return annotation_prompt


def get_flat_annotation_prompt(flat_taxonomy_str, job_posting_str):
    annotation_prompt = f"""You are an expert job classification system. Your task is to analyze the provided job posting and assign it to the most appropriate occupation(s) from the given flat taxonomy.

Carefully review the entire job posting and the provided taxonomy. The taxonomy is a flat list of occupations. Each occupation has a unique ID (e.g., L0.1, L0.2), a title (name), and a description. All occupations are at the same level (L0).

**Annotation Rules:**

1.  **Primary Goal:** Identify the single most appropriate occupation from the taxonomy that best describes the core responsibilities and nature of the job posting. You must choose from the provided L0 occupations.
2.  **Granularity:** Since the taxonomy is flat, you will directly assign the job posting to the L0 occupation that accurately matches its content.
3.  **Multiple Occupations (Use Sparingly):**
    *   Ideally, assign only one occupation.
    *   However, if the job posting genuinely and substantially describes responsibilities that clearly map to *two or more distinct L0 occupations* within the taxonomy, and a single occupation would be insufficient, you may list multiple occupations.
    *   If listing multiple, ensure each choice is a strong and distinct fit.
4.  **"Other" Category:** If, after careful consideration, no L0 occupation in the provided taxonomy is a suitable fit for the job posting, you will indicate this using the `is_other` flag.

**Output Format:**

Provide your response as a JSON object with the following structure:

```json
{{{{
  "assigned_occupations": [
    // This array will contain occupation objects if a match is found.
    // Example: {{"id": "L0.X", "title": "Title of the assigned occupation"}}
    // This array will be EMPTY if no match is found (and is_other is true).
  ],
  "is_other": true/false // true if no suitable occupation is found, false otherwise.
}}}}

Example of assigned_occupations array entries:
Single assignment: {{{{
  "assigned_occupations": [{{"id": "L0.15", "title": "Grounds and Landscape Maintenance Worker"}}],
  "is_other": false
}}}}
Multiple assignments (extremely rare): {{{{
  "assigned_occupations": [
    {{"id": "L0.17", "title": "IT Network and Support Specialist"}},
    {{"id": "L0.6", "title": "Business Systems Analyst"}} // If job truly spans both distinctly
  ],
  "is_other": false
}}}}
Other: {{{{
  "assigned_occupations": [], // Empty array
  "is_other": true
}}}}

---START OF TAXONOMY---
{flat_taxonomy_str}
---END OF TAXONOMY---

---START OF JOB POSTING---
{job_posting_str}
---END OF JOB POSTING---

Your response MUST be a single, valid JSON object and nothing else. Do not include any explanatory text or markdown formatting before or after the JSON object.
Now, please provide the JSON output for the job posting above.
"""
    return annotation_prompt

def get_esco_annotation_prompt(job_posting_str, taxonomy_str):
    annotation_prompt = f"""You are an expert job classification system. Your task is to analyze the provided job posting and assign it to the most appropriate occupation(s) from the given hierarchical taxonomy.

Carefully review the entire job posting and the provided ESCO/ISCO taxonomy. The taxonomy is structured hierarchically, with unique IDs for each occupation (e.g., L0.1, L1.1.1). Leaf occupation means any node in the tree that has no children. Each occupation, whether a broad category (L0) or a specific leaf role (L1), has a title and a description.

**Annotation Rules:**

1.  **Primary Goal:** Identify the single most appropriate occupation from the taxonomy that best describes the core responsibilities and nature of the job posting.
2.  **Granularity:**
    *   If the job posting provides enough specific detail, assign it to the most granular leaf occupation that accurately matches.
    *   If the job posting is too vague for a specific leaf occupation but clearly aligns with a broader parent occupation (L0), assign it to that parent occupation. Choose the most specific parent occupation that is still a good fit.
3.  **Multiple Occupations (Use VERY Sparingly):**
    *   Ideally, assign only one occupation.
    *   However, if the job posting genuinely and substantially describes responsibilities that clearly map to *two or more distinct occupations* within the taxonomy, and a single occupation would be insufficient, you may list multiple occupations.
    *   If listing multiple, ensure each choice is a strong fit. Do not list a parent and its child simultaneously.
4.  **"Other" Category:** If, after careful consideration, no occupation in the provided taxonomy (at any level) is a suitable fit for the job posting, assign it as "Other".

**Output Format:**

Provide your response as a JSON object with the following structure:

```json
{{{{
"assigned_occupations": [
    // This array will contain occupation objects if a match is found.
    // Example: {{"id": "L0.X", "title": "Title of the assigned occupation"}}
    // This array will be EMPTY if no match is found (and is_other is true).
],
"is_other": true/false // true if no suitable occupation is found, false otherwise.
}}}}

Example of assigned_occupations array entries:
Single leaf assignment: {{{{
"assigned_occupations": [{{"id": "L0.1", "title": "Care Coordinator"}}],
"is_other": false
}}}}
Single parent assignment: {{{{
"assigned_occupations": [{{"id": "L1.1.2", "title": "Healthcare Direct Care and Social Services"}}],
"is_other": false
}}}}
Multiple assignments (extremely rare): {{{{
"assigned_occupations": [
    {{"id": "L0.X", "title": "Specific Role A"}},
    {{"id": "L0.A", "title": "Specific Role B"}}
],
"is_other": false
}}}}
Other: {{{{
"assigned_occupations": [], // Empty array
"is_other": true
}}}}

---START OF TAXONOMY---
{taxonomy_str}
---END OF TAXONOMY---

---START OF JOB POSTING---
{job_posting_str}
---END OF JOB POSTING---

Your response MUST be a single, valid JSON object and nothing else. Do not include any explanatory text or markdown formatting before or after the JSON object.
Now, please provide the JSON output for the job posting above.
"""
    return annotation_prompt

# --- Main Workflow Functions ---

def load_data_files(
    test_data_path: str = 'indeed_test_data_1165.pkl', 
    tree_taxonomy_file: str = 'full_generated_taxonomy_with_desc.json',
    flat_taxonomy_file: str = 'tnt_clusters_full.pkl'
) -> Dict[str, Any]:
    """Loads all necessary data files."""
    print("Loading data files...")
    data = {}
    try:
        with open(DATA_DIR / "esco" / "en" / "occupations_en.csv", "r", encoding="utf-8") as f:
            data["esco_occupations_csv"] = f.read()
        with open(DATA_DIR / "esco" / "en" / "ISCOGroups_en.csv", "r", encoding="utf-8") as f:
            data["isco_groups_csv"] = f.read()
        with open(tree_taxonomy_file, "r", encoding="utf-8") as f:
            json_string = f.read()
            # data["tree_taxonomy_json"] = json.load(f)
            taxonomy_data_loaded = json.loads(json_string)
            taxonomy_data_loaded = {k:v for k,v in taxonomy_data_loaded.items() if int(k)<3}
            data["tree_taxonomy_json"] = taxonomy_data_loaded
        with open(flat_taxonomy_file, "rb") as f:
            # Expects a list of {"title": "...", "description": "..."}
            data["flat_taxonomy_items"] = pickle.load(f)
        # with open(DATA_DIR / "sample_job_posting.txt", "r", encoding="utf-8") as f: #TODO: load the job posting from the file
        #     data["job_posting"] = f.read()
        # df_train = pickle.load(open(DATA_DIR / 'indeed_train_data_10129.pkl', 'rb'))
        df_test = pickle.load(open(DATA_DIR / test_data_path, 'rb'))
        print(len(df_test))
        data["job_posting"] = df_test

        with open("title2desc.pkl", "rb") as f:
            data["title2desc"] = pickle.load(f)
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure all data files are in the '{DATA_DIR}' directory.")
        return {}
    return data

def process_and_reformat_all_taxonomies(
    loaded_data: Dict[str, Any], 
    tree_taxonomy_file: str,
    flat_taxonomy_file: str,
    cache_prefix: str = "",
    tree_desc_len: int = 50,
    permute_branches: bool = False,
    enabled_taxonomies: Optional[List[str]] = None
) -> Dict[str, str]:
    """Processes and reformats all loaded taxonomies."""
    if not loaded_data:
        return {}
    
    taxonomies = {}
        
    if not enabled_taxonomies or "esco" in enabled_taxonomies:
        if not os.path.exists(DATA_DIR / "esco_id_to_title_map.json"):
            print("Reformatting ESCO taxonomy...")
            esco_taxonomy_str, esco_id_to_code_map, _, _= reformat_esco_under_isco_tree(
                loaded_data["esco_occupations_csv"],
                loaded_data["isco_groups_csv"],
                min_level=3,  # Example: ISCO Sub-Major Group and deeper
                max_level=4,   # Example: ESCO occupations with one dot
                desc_len=20
            )
            # save esco_id_to_title_map to a json file
            with open(DATA_DIR / "esco_id_to_code_map.json", "w", encoding="utf-8") as f:
                json.dump(esco_id_to_code_map, f, ensure_ascii=False)
            # save esco_taxonomy_str to a txt file
            #pickle the esco_taxonomy_str
            with open(DATA_DIR / "esco_taxonomy_str.pkl", "wb") as f:
                pickle.dump(esco_taxonomy_str, f)
        else:
            print("Loading ESCO taxonomy from cache...")
            with open(DATA_DIR / "esco_id_to_code_map.json", "r", encoding="utf-8") as f:
                esco_id_to_code_map = json.load(f)
            with open(DATA_DIR / "esco_taxonomy_str.pkl", "rb") as f:
                esco_taxonomy_str = pickle.load(f)
        esco_taxonomy_str, esco_id_to_code_map, _, _= reformat_esco_under_isco_tree(
            loaded_data["esco_occupations_csv"],
            loaded_data["isco_groups_csv"],
            min_level=3,  # Example: ISCO Sub-Major Group and deeper
            max_level=4,   # Example: ESCO occupations with one dot
            desc_len=20
        )
        print(len(esco_taxonomy_str.split(' ')))
        # print(esco_taxonomy_str[:500])
        taxonomies["esco"] = esco_taxonomy_str

    if not enabled_taxonomies or "tree" in enabled_taxonomies:
        prefix = f"{cache_prefix}_" if cache_prefix else ""
        tree_taxonomy_stem = Path(tree_taxonomy_file).stem
        
        desc_len_suffix = f"desc{'full' if tree_desc_len < 0 else tree_desc_len}"
        permute_suffix = "_permuted" if permute_branches else ""
        tree_id_map_path = DATA_DIR / f"{prefix}{tree_taxonomy_stem}_{desc_len_suffix}{permute_suffix}_id_to_title_map.json"
        tree_str_path = DATA_DIR / f"{prefix}{tree_taxonomy_stem}_{desc_len_suffix}{permute_suffix}_taxonomy_str.pkl"

        if not tree_id_map_path.exists() or not tree_str_path.exists():
            print(f"Reformatting Tree taxonomy from {tree_taxonomy_file}...")
            tree_taxonomy_str, tree_id_to_title_map = reformat_tree_taxonomy_for_llm(
                loaded_data["tree_taxonomy_json"], 
                loaded_data["title2desc"], 
                desc_len=tree_desc_len,
                permute_branches=permute_branches
            )
            # save tree_id_to_title_map to a json file
            with open(tree_id_map_path, "w", encoding="utf-8") as f:
                json.dump(tree_id_to_title_map, f, ensure_ascii=False)
            # save tree_taxonomy_str to a txt file
            with open(tree_str_path, "wb") as f:
                pickle.dump(tree_taxonomy_str, f)
        else:
            print(f"Loading Tree taxonomy from cache for {tree_taxonomy_file}...")
            with open(tree_id_map_path, "r", encoding="utf-8") as f:
                tree_id_to_title_map = json.load(f)
            with open(tree_str_path, "rb") as f:
                tree_taxonomy_str = pickle.load(f)
        print(len(tree_taxonomy_str.split(' ')))
        # print(tree_taxonomy_str[:500])
        taxonomies["tree"] = tree_taxonomy_str

    if not enabled_taxonomies or "flat" in enabled_taxonomies:
        prefix = f"{cache_prefix}_" if cache_prefix else ""
        flat_taxonomy_stem = Path(flat_taxonomy_file).stem
        flat_id_map_path = DATA_DIR / f"{prefix}{flat_taxonomy_stem}_id_to_title_map.json"
        flat_str_path = DATA_DIR / f"{prefix}{flat_taxonomy_stem}_taxonomy_str.pkl"

        if not flat_id_map_path.exists() or not flat_str_path.exists():
            print(f"Reformatting Flat taxonomy from {flat_taxonomy_file}...")
            flat_taxonomy_str, flat_id_to_title_map = reformat_flat_taxonomy_for_llm(loaded_data["flat_taxonomy_items"], desc_len=400)
            # save flat_id_to_title_map to a json file
            with open(flat_id_map_path, "w", encoding="utf-8") as f:
                json.dump(flat_id_to_title_map, f, ensure_ascii=False)
            with open(flat_str_path, "wb") as f:
                pickle.dump(flat_taxonomy_str, f)
        else:
            print(f"Loading Flat taxonomy from cache for {flat_taxonomy_file}...")
            with open(flat_id_map_path, "r", encoding="utf-8") as f:
                flat_id_to_title_map = json.load(f)
            with open(flat_str_path, "rb") as f:
                flat_taxonomy_str = pickle.load(f)
        print(len(flat_taxonomy_str.split(' ')))
        # print(flat_taxonomy_str[:500])
        taxonomies["flat"] = flat_taxonomy_str

    return taxonomies

def run_single_llm_annotations(
    job_posting_str: str,
    formatted_taxonomies: Dict[str, str],
    job_id: str,
    target_dir: Path,
    llm_client: Any, # The actual client object
    llm_model_name: str,
    llm_api_call_function: Callable, # e.g., call_openai_api
    llm_display_name: str, # For filenames and keys, e.g., "openai_gpt4o"
    temperature: float,
    max_tokens: int
) -> Dict[str, Any]:
    """
    Runs annotations for a single specified LLM across all provided taxonomies.
    Saves each result individually, creating a file for both successes and failures.
    Returns a dictionary of the successful results.
    """
    results = {}
    target_dir.mkdir(parents=True, exist_ok=True)

    if not llm_client:
        print(f"LLM client for '{llm_model_name}' not provided. Skipping and logging failure for this LLM.")
        for tax_type in formatted_taxonomies.keys():
            reason = "LLM client not configured or enabled"
            safe_llm_name = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in llm_model_name)
            safe_tax_type = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in tax_type)
            output_filename = target_dir / f"{job_id}_{safe_llm_name}_{safe_tax_type}_annotation.json"
            failure_payload = {
                "annotation": {"status": "FAILED", "reason": reason},
                "job_id": job_id, "llm": llm_display_name,
                "model_used": llm_model_name, "taxonomy_type": tax_type
            }
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(failure_payload, f, indent=4)
        return results

    for tax_type, tax_content in formatted_taxonomies.items():
        if not tax_content:
            print(f"Skipping taxonomy '{tax_type}' for job_id '{job_id}', LLM '{llm_model_name}' as its content is empty.")
            continue

        result_key = f"{llm_model_name}_{tax_type}"
        safe_llm_name = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in llm_model_name)
        safe_tax_type = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in tax_type)
        output_filename = target_dir / f"{job_id}_{safe_llm_name}_{safe_tax_type}_annotation.json"

        current_llm_result = None
        if output_filename.exists():
            try:
                with open(output_filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                raw_annotation = data.get("annotation")
                # If it's a dict and marked as FAILED, we must re-run it.
                if isinstance(raw_annotation, dict) and raw_annotation.get("status") == "FAILED":
                    print(f"Found previous failed annotation for {output_filename.name}. Re-running.")
                    current_llm_result = None
                # Otherwise, check if it's a valid, completed annotation
                elif data.get("llm") == llm_display_name and data.get("model_used") == llm_model_name:
                    current_llm_result = raw_annotation
                    print(f"Loaded existing successful annotation for job '{job_id}' from: {output_filename.name}")
                else:
                    print(f"Warning: File {output_filename.name} has non-matching metadata. Re-running.")
            except Exception as e:
                print(f"Error loading or parsing {output_filename.name}, re-running: {e}")
                current_llm_result = None

        if current_llm_result is None:
            if tax_type == "flat":
                prompt = get_flat_annotation_prompt(tax_content, job_posting_str)
            elif tax_type == "tree":
                prompt = get_tree_annotation_prompt(tax_content, job_posting_str)
            elif tax_type == "esco":
                prompt = get_esco_annotation_prompt(job_posting_str, tax_content)
            else:
                raise ValueError(f"Invalid taxonomy type: {tax_type}")
            
            print(f"Calling {llm_display_name} (model: {llm_model_name}) for job '{job_id}', taxonomy '{tax_type}'...")
            
            api_response_content = llm_api_call_function(
                llm_client, prompt, llm_model_name, temperature, max_tokens
            )
            
            is_error = not api_response_content or (isinstance(api_response_content, str) and api_response_content.startswith("API_ERROR:"))
            
            payload_to_save = {
                "job_id": job_id, "llm": llm_display_name,
                "model_used": llm_model_name, "taxonomy_type": tax_type
            }

            if not is_error:
                current_llm_result = api_response_content
                payload_to_save["annotation"] = current_llm_result
                print(f"Saved successful annotation for job '{job_id}' to: {output_filename.name}")
            else:
                reason = api_response_content if api_response_content else "API call returned no content"
                payload_to_save["annotation"] = {"status": "FAILED", "reason": reason}
                print(f"Warning: API call failed for job '{job_id}'. Saving failure record to {output_filename.name}.")

            try:
                with open(output_filename, "w", encoding="utf-8") as f:
                    json.dump(payload_to_save, f, indent=4)
            except Exception as e:
                print(f"CRITICAL: Could not save annotation file {output_filename.name}: {e}")

        if current_llm_result:
            results[result_key] = current_llm_result
            
    return results


def run_bulk_annotations(
    df_job_postings: pd.DataFrame, 
    formatted_taxonomies: Dict[str, str], 
    bulk_output_dir: Path,
    llm_configurations: List[Dict[str, Any]],
    use_column: str = 'text'
):
    """
    Runs annotations for all job postings in the DataFrame using configured LLMs.
    It iterates through LLM configurations and calls run_single_llm_annotations for each.
    """
    print(f"\nStarting bulk annotation process. Individual results will be saved in: {bulk_output_dir}")
    bulk_output_dir.mkdir(parents=True, exist_ok=True)

    if not llm_configurations:
        print("No LLM configurations provided for bulk annotation. Aborting.")
        return

    total_jobs = len(df_job_postings)

    for index, job_row in tqdm(df_job_postings.iterrows(), total=total_jobs, desc="Processing Job Postings"):
        job_id_val = job_row.get('job_id')
        if pd.isna(job_id_val) or str(job_id_val).strip() == '':
             job_id = f"index_{index}"
        else:
             job_id = str(job_id_val).strip()
        
        job_title = job_row.get('job_title', '')
        job_text = job_row.get(use_column, '') 
        
        if not job_text:
            print(f"Warning: Skipping job_id: {job_id} (Job {index + 1}/{total_jobs}) due to missing or empty job text/description.")
            continue

        job_posting_str = f"{job_title}\n{job_text}".strip()
        
        print(f"\nAnnotating job_id: {job_id} (Job {index + 1}/{total_jobs}) with configured LLMs...")
        
        for llm_config in llm_configurations:
            if not llm_config.get("enabled", False) or not llm_config.get("client"):
                if llm_config.get("name"):
                    print(f"Skipping LLM '{llm_config['name']}' for job '{job_id}' (disabled or client not configured).")
                else:
                    print(f"Skipping an LLM configuration for job '{job_id}' (incomplete config).")
                continue

            llm_display_name = llm_config.get("display_name", llm_config["name"])

            print(f"---> Using LLM: {llm_display_name} for job {job_id}")
            run_single_llm_annotations(
                job_posting_str=job_posting_str,
                formatted_taxonomies=formatted_taxonomies,
                job_id=job_id,
                target_dir=bulk_output_dir,
                llm_client=llm_config["client"],
                llm_model_name=llm_config["model_name"],
                llm_api_call_function=llm_config["api_call_function"],
                llm_display_name=llm_display_name,
                temperature=llm_config["temperature"],
                max_tokens=llm_config["max_tokens"]
            )

    print("\nBulk annotation process finished.")
    print(f"Total jobs processed: {total_jobs}")
    print(f"Individual annotation files are in: {bulk_output_dir}")

def get_ancestor(label_id: str, target_depth: int) -> Optional[str]:
    """
    Given a label ID (e.g., 'L1.5.2'), returns its ancestor at a specific depth.
    Depth 0 is the root (e.g., 'L1.5').
    Depth 1 is the next level down (e.g., 'L1.5.2').
    """
    if not isinstance(label_id, str) or '.' not in label_id:
        return label_id
        
    parts = label_id.split('.')
    # The number of path segments is depth + 2 (e.g., depth 0 is ['L1', '5'])
    num_parts_to_keep = target_depth + 2
    
    if len(parts) < num_parts_to_keep:
        return label_id # Return the full ID if requested depth is deeper than the node
        
    return ".".join(parts[:num_parts_to_keep])

def calculate_hierarchical_distance(id1: str, id2: str) -> Optional[int]:
    """
    Calculates the hierarchical distance between two taxonomy IDs based on their
    paths from their lowest common ancestor.
    Distance = (depth(id1) - depth(LCA)) + (depth(id2) - depth(LCA))
    """
    if not id1 or not id2 or not isinstance(id1, str) or not isinstance(id2, str):
        return None  # Cannot compare invalid IDs

    parts1 = id1.split('.')
    parts2 = id2.split('.')

    # Find the length of the path to the lowest common ancestor
    lca_path_len = 0
    for i in range(min(len(parts1), len(parts2))):
        if parts1[i] == parts2[i]:
            lca_path_len += 1
        else:
            break
            
    # Depth is measured from the root of the sub-tree (e.g., L0.1 is depth 0)
    depth1 = len(parts1) - 2 
    depth2 = len(parts2) - 2
    lca_depth = lca_path_len - 2

    if lca_depth < -1: # Should not happen with valid IDs like L0.1
        return None

    # The distance is the sum of steps from each node up to the LCA
    distance = (depth1 - lca_depth) + (depth2 - lca_depth)
    return distance

def load_and_parse_annotations(bulk_output_dir: Path) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Loads all annotation JSON files, parsing them for evaluation
    and preparing a detailed list of rows for a CSV report.
    """
    all_annotations = {}
    dataframe_rows = []
    
    if not bulk_output_dir.exists():
        print(f"Annotation directory not found: {bulk_output_dir}")
        return {}, []

    annotation_files = list(bulk_output_dir.glob("*_annotation.json"))
    print(f"Found {len(annotation_files)} annotation files in {bulk_output_dir}")

    for file_path in annotation_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            job_id = data.get("job_id")
            # Use the specific model_name for evaluation, fallback to the display name, and format it.
            model_name_raw = data.get("model_used") or data.get("llm")
            
            llm = model_name_raw
            if llm:
                if '/' in llm:
                    llm = llm.split('/')[-1]
                if ':free' in llm:
                    llm = llm.replace(':free', '')
                    
            tax_type = data.get("taxonomy_type")
            raw_annotation = data.get("annotation")

            if not all([job_id, llm, tax_type, raw_annotation]):
                print(f"Warning: Skipping file {file_path.name} due to missing core metadata.")
                continue
            
            annotation_json = None
            row_to_add = { "job_id": job_id, "llm": llm, "taxonomy_type": tax_type }

            # Check for failure marker or parse the successful annotation string
            if isinstance(raw_annotation, dict) and raw_annotation.get("status") == "FAILED":
                print(f"Info: Treating failed annotation as 'Other' for evaluation: {file_path.name}")
                annotation_json = {"assigned_occupations": [], "is_other": True}
                row_to_add.update({
                    "is_other": True, "assigned_ids": "", "assigned_titles": "",
                    "status": "FAILED", "reason": raw_annotation.get("reason", "Unknown")
                })

            elif isinstance(raw_annotation, str):
                annotation_str = raw_annotation
                try:
                    if annotation_str.strip().startswith("```json"):
                        annotation_str = annotation_str.strip()[7:-3].strip()
                    elif annotation_str.strip().startswith("```"):
                        annotation_str = annotation_str.strip()[3:-3].strip()
                    
                    annotation_json = json.loads(annotation_str)
                    is_other = annotation_json.get("is_other", False)
                    assigned_occupations = annotation_json.get("assigned_occupations", [])
                    assigned_ids = "|".join(sorted([str(occ.get('id', '')) for occ in assigned_occupations])) if not is_other else ""
                    assigned_titles = "|".join(sorted([str(occ.get('title', '')) for occ in assigned_occupations])) if not is_other else ""
                    
                    row_to_add.update({
                        "is_other": is_other, "assigned_ids": assigned_ids, "assigned_titles": assigned_titles,
                        "status": "SUCCESS", "reason": ""
                    })

                except json.JSONDecodeError:
                    print(f"Warning: Could not parse annotation JSON in {file_path.name}. Treating as 'Other'.")
                    annotation_json = {"assigned_occupations": [], "is_other": True}
                    row_to_add.update({
                        "is_other": True, "assigned_ids": "", "assigned_titles": "",
                        "status": "FAILED", "reason": "JSONDecodeError"
                    })
            else:
                print(f"Warning: Skipping file {file_path.name} due to malformed 'annotation' field.")
                continue

            dataframe_rows.append(row_to_add)

            if job_id not in all_annotations:
                all_annotations[job_id] = {}
            if tax_type not in all_annotations[job_id]:
                all_annotations[job_id][tax_type] = {}
            
            all_annotations[job_id][tax_type][llm] = annotation_json

        except Exception as e:
            print(f"Error processing file {file_path.name}: {e}")
            
    return all_annotations, dataframe_rows

def evaluate_annotations(
    bulk_output_dir: Path, 
    dataframe_rows: List[Dict[str, Any]],
    ground_truth_df: Optional[pd.DataFrame] = None,
    id_to_title_maps: Optional[Dict[str, Dict]] = None
):
    """
    Calculates and prints evaluation metrics for LLM annotations.
    - Mean Coverage (as mean "other" rate)
    - Pairwise Agreement Ratio
    """
    
    # 1. Load and parse all annotations
    # This is now done outside and the results are passed in.
    # We still need to build the nested dict for metric calculations.
    all_annotations = {}
    for row in dataframe_rows:
        job_id, tax_type, llm = row['job_id'], row['taxonomy_type'], row['llm']
        if job_id not in all_annotations:
            all_annotations[job_id] = {}
        if tax_type not in all_annotations[job_id]:
            all_annotations[job_id][tax_type] = {}
        
        # Reconstruct the annotation object needed for evaluation from the row
        all_annotations[job_id][tax_type][llm] = {
            "is_other": row['is_other'],
            "assigned_occupations": [
                {"id": id, "title": title} 
                for id, title in zip(row['assigned_ids'].split('|'), row['assigned_titles'].split('|'))
                if id # check for empty string
            ]
        }
    
    if not dataframe_rows:
        print("No valid annotations found to evaluate.")
        return

    # Create a flat list to hold data for the DataFrame - THIS IS NOW PASSED IN
    # dataframe_rows = []

    # 2. Structure data for easier analysis: { tax_type -> { llm -> { job_id -> annotation } } }
    annotations_by_tax = {}
    all_llms = set()
    all_tax_types = set()

    for job_id, tax_data in all_annotations.items():
        for tax_type, llm_data in tax_data.items():
            all_tax_types.add(tax_type)
            if tax_type not in annotations_by_tax:
                annotations_by_tax[tax_type] = {}
            for llm, annotation in llm_data.items():
                all_llms.add(llm)
                if llm not in annotations_by_tax[tax_type]:
                    annotations_by_tax[tax_type][llm] = {}
                annotations_by_tax[tax_type][llm][job_id] = annotation

    # Create and save the DataFrame from the collected rows
    if dataframe_rows:
        results_df = pd.DataFrame(dataframe_rows)
        # Reorder columns for better readability
        cols_order = [
            "job_id", "llm", "taxonomy_type", "status", "reason", 
            "is_other", "assigned_ids", "assigned_titles"
        ]
        results_df = results_df[[c for c in cols_order if c in results_df.columns]]
        
        output_csv_path = bulk_output_dir.parent / f"{bulk_output_dir.name}_full_results.csv"
        try:
            results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
            print(f"\nSaved detailed annotation results to: {output_csv_path}")
        except Exception as e:
            print(f"Error saving results DataFrame to CSV: {e}")

    # Determine the maximum length for aligned printing
    max_llm_name_len = 0
    if all_llms:
        max_llm_name_len = max(len(llm) for llm in all_llms)

    print(f"\nFound annotations for taxonomies: {sorted(list(all_tax_types))}")
    print(f"Found annotations from LLMs: {sorted(list(all_llms))}")
    print("-" * 20)
    
    # --- NEW: Failure Analysis ---
    print("\nFailure Analysis (by LLM and Taxonomy):")
    failure_counts = Counter()
    total_counts = Counter()
    for row in dataframe_rows:
        key = (row['llm'], row['taxonomy_type'])
        total_counts[key] += 1
        if row['status'] == 'FAILED':
            failure_counts[key] += 1
    
    # Restructure for printing
    failures_by_tax_summary = {}
    all_tax_types_from_rows = sorted(list({row['taxonomy_type'] for row in dataframe_rows}))

    for tax_type in all_tax_types_from_rows:
        print(f"\n--- Taxonomy: {tax_type.upper()} ---")
        llms_in_tax = sorted(list({row['llm'] for row in dataframe_rows if row['taxonomy_type'] == tax_type}))
        for llm in llms_in_tax:
            key = (llm, tax_type)
            fail_count = failure_counts.get(key, 0)
            total_count = total_counts.get(key, 0)
            if total_count > 0:
                fail_rate = fail_count / total_count
                print(f"  - {llm:>{max_llm_name_len}}: {fail_count} failures out of {total_count} attempts ({fail_rate:.2%})")
            else:
                print(f"  - {llm:>{max_llm_name_len}}: No annotation attempts found.")
    
    # --- Taxonomy Utilization Analysis ---
    print("\n\nTaxonomy Utilization Rate Analysis:")
    if not id_to_title_maps:
        print("  Skipping: ID-to-Title maps are not available for calculating utilization.")
    else:
        for tax_type in sorted(annotations_by_tax.keys()):
            print(f"\n--- Taxonomy: {tax_type.upper()} ---")

            total_labels_in_taxonomy = len(id_to_title_maps.get(tax_type, {}))
            if total_labels_in_taxonomy == 0:
                print("  Total labels in taxonomy is 0. Cannot calculate utilization.")
                continue
            
            print(f"  (Total labels available in this taxonomy: {total_labels_in_taxonomy})")

            llms_in_tax = sorted(list(annotations_by_tax[tax_type].keys()))
            for llm in llms_in_tax:
                used_labels = set()
                for annotation in annotations_by_tax[tax_type][llm].values():
                    if not annotation.get("is_other", False):
                        for occ in annotation.get("assigned_occupations", []):
                            if occ.get('id'):
                                used_labels.add(occ['id'])
                
                num_used_labels = len(used_labels)
                utilization_rate = num_used_labels / total_labels_in_taxonomy
                
                print(f"  - {llm:>{max_llm_name_len}}: Used {num_used_labels} unique labels ({utilization_rate:.2%})")

    # --- 3. Calculate Mean Coverage ("Other" Rate) ---
    print("\n\nCoverage Analysis (Mean 'Other' Rate):")
    
    for tax_type in sorted(annotations_by_tax.keys()):
        print(f"\n--- Taxonomy: {tax_type.upper()} ---")
        llm_other_rates = {}
        
        for llm in sorted(annotations_by_tax[tax_type].keys()):
            annotations = annotations_by_tax[tax_type][llm]
            total_samples = len(annotations)
            if total_samples == 0:
                continue

            other_count = sum(1 for ann in annotations.values() if ann.get("is_other", False))
            other_rate = other_count / total_samples
            coverage = 1 - other_rate
            llm_other_rates[llm] = other_rate
            
            print(f"  - LLM: {llm:>{max_llm_name_len}} | Coverage: {coverage:.2%} ({total_samples - other_count}/{total_samples} labeled) | 'Other' Rate: {other_rate:.2%}")

        if len(llm_other_rates) > 1:
            mean_other_rate = sum(llm_other_rates.values()) / len(llm_other_rates)
            print(f"  -------------------------------------")
            print(f"  Mean 'Other' Rate across {len(llm_other_rates)} LLMs: {mean_other_rate:.2%}")

    # --- 4. Calculate Agreement Ratios ---
    print("\n\nAgreement Analysis (Strict: Agreement requires a specific, identical label. 'Other' is always a disagreement.):")

    for tax_type in sorted(annotations_by_tax.keys()):
        print(f"\n--- Taxonomy: {tax_type.upper()} ---")
        
        if id_to_title_maps and tax_type in id_to_title_maps:
            total_labels = len(id_to_title_maps[tax_type])
            print(f"  (Total labels available in this taxonomy: {total_labels})")
        
        llms_for_tax = sorted(list(annotations_by_tax[tax_type].keys()))
        if len(llms_for_tax) < 2:
            print("  Not enough LLMs for agreement comparison.")
            continue

        # --- Overall N-Way Agreement (all LLMs must agree) ---
        if len(llms_for_tax) > 1:
            print(f"  --- Overall Agreement ({len(llms_for_tax)} LLMs) ---")
            
            # Find common job IDs across all LLMs for this taxonomy
            try:
                common_job_ids = set.intersection(*[set(annotations_by_tax[tax_type][llm].keys()) for llm in llms_for_tax])
            except KeyError:
                common_job_ids = set()

            if not common_job_ids:
                print("    No common jobs found across all LLMs to compare for N-way agreement.")
            else:
                n_way_agreement_count = 0
                comparison_count = len(common_job_ids) # Denominator is all common jobs

                for job_id in common_job_ids:
                    # Get the annotation from the first LLM to use as the reference
                    first_ann = annotations_by_tax[tax_type][llms_for_tax[0]][job_id]
                    
                    # If the reference is 'Other' or has no labels, it cannot be an agreement.
                    if first_ann.get("is_other", False):
                        continue
                    labels_ref = {occ.get('id') for occ in first_ann.get("assigned_occupations", []) if occ.get('id')}
                    if not labels_ref:
                        continue

                    all_agree = True
                    # Compare with the rest of the LLMs
                    for i in range(1, len(llms_for_tax)):
                        other_ann = annotations_by_tax[tax_type][llms_for_tax[i]][job_id]
                        # Any 'Other' in the comparison group results in disagreement
                        if other_ann.get("is_other", False):
                            all_agree = False
                            break
                        labels_comp = {occ.get('id') for occ in other_ann.get("assigned_occupations", []) if occ.get('id')}
                        if labels_ref != labels_comp:
                            all_agree = False
                            break
                    
                    if all_agree:
                        n_way_agreement_count += 1
                
                agreement_ratio = n_way_agreement_count / comparison_count
                print(f"    - All {len(llms_for_tax)} LLMs agree on: {agreement_ratio:.2%} ({n_way_agreement_count}/{comparison_count} common jobs)")

        # --- Pairwise Agreement ---
        print(f"\n  --- Pairwise Agreement ---")
        try:
            from sklearn.metrics import cohen_kappa_score
            kappa_available = True
        except ImportError:
            kappa_available = False
            print("    Skipping Cohen's Kappa calculation: `scikit-learn` is not installed. Run `pip install scikit-learn` to enable.")

        for llm1, llm2 in combinations(llms_for_tax, 2):
            agreement_count = 0
            
            ann1_map = annotations_by_tax[tax_type][llm1]
            ann2_map = annotations_by_tax[tax_type][llm2]
            
            common_job_ids = set(ann1_map.keys()) & set(ann2_map.keys())
            comparison_count = len(common_job_ids)
            
            labels1_for_kappa = []
            labels2_for_kappa = []

            for job_id in common_job_ids:
                ann1 = ann1_map[job_id]
                ann2 = ann2_map[job_id]

                is_other1 = ann1.get("is_other", False)
                is_other2 = ann2.get("is_other", False)

                # Define categories for kappa calculation.
                # The category is the frozenset of assigned IDs, or a special string for "Other"/"No Label"
                cat1 = "Other" if is_other1 else (frozenset({occ.get('id') for occ in ann1.get("assigned_occupations", []) if occ.get('id')}) or "No Label Given")
                cat2 = "Other" if is_other2 else (frozenset({occ.get('id') for occ in ann2.get("assigned_occupations", []) if occ.get('id')}) or "No Label Given")
                
                # Append string representation of categories for kappa
                labels1_for_kappa.append(str(cat1))
                labels2_for_kappa.append(str(cat2))

                # Agreement only if NEITHER is 'Other' and their non-empty labels match.
                if not is_other1 and not is_other2:
                    labels1 = {occ.get('id') for occ in ann1.get("assigned_occupations", []) if occ.get('id')}
                    labels2 = {occ.get('id') for occ in ann2.get("assigned_occupations", []) if occ.get('id')}
                    if labels1 and labels1 == labels2:
                        agreement_count += 1
            
            if comparison_count > 0:
                agreement_ratio = agreement_count / comparison_count
                print(f"  - Agreement between {llm1:>{max_llm_name_len}} and {llm2:>{max_llm_name_len}}: {agreement_ratio:.2%} ({agreement_count}/{comparison_count} agreements)")

                if kappa_available and len(labels1_for_kappa) > 0:
                    try:
                        kappa = cohen_kappa_score(labels1_for_kappa, labels2_for_kappa)
                        interp = "Poor"
                        if kappa > 0.8: interp = "Almost Perfect"
                        elif kappa > 0.6: interp = "Substantial"
                        elif kappa > 0.4: interp = "Moderate"
                        elif kappa > 0.2: interp = "Fair"
                        elif kappa > 0.0: interp = "Slight"
                        print(f"    - Cohen's Kappa: {kappa:.4f} ({interp} agreement)")
                    except Exception as e:
                        print(f"    - Could not calculate Cohen's Kappa: {e}")
            else:
                print(f"  - No overlapping jobs found for {llm1} and {llm2} to compare.")

        # --- Fleiss' Kappa ---
        print(f"\n  --- Fleiss' Kappa Analysis ({len(llms_for_tax)} LLMs) ---")
        try:
            from statsmodels.stats.inter_rater import fleiss_kappa
            import numpy as np

            common_job_ids = sorted(list(set.intersection(*[set(annotations_by_tax[tax_type][llm].keys()) for llm in llms_for_tax])))
            
            if not common_job_ids or len(llms_for_tax) < 2:
                print("    Not enough common data to calculate Fleiss' Kappa.")
                continue

            all_categories = set()
            for job_id in common_job_ids:
                for llm in llms_for_tax:
                    ann = annotations_by_tax[tax_type][llm][job_id]
                    if ann.get("is_other", False):
                        category = "Other"
                    else:
                        labels = {occ.get('id') for occ in ann.get("assigned_occupations", []) if occ.get('id')}
                        if not labels:
                            category = "No Label Given"
                        else:
                            category = frozenset(labels)
                    all_categories.add(category)
            
            category_list = sorted(list(all_categories), key=lambda x: str(x))
            category_to_idx = {cat: i for i, cat in enumerate(category_list)}
            
            num_subjects = len(common_job_ids)
            num_categories = len(category_list)
            
            ratings_matrix = np.zeros((num_subjects, num_categories), dtype=int)
            job_id_to_idx = {job_id: i for i, job_id in enumerate(common_job_ids)}

            for llm in llms_for_tax:
                for job_id in common_job_ids:
                    row_idx = job_id_to_idx[job_id]
                    ann = annotations_by_tax[tax_type][llm][job_id]
                    
                    if ann.get("is_other", False):
                        category = "Other"
                    else:
                        labels = {occ.get('id') for occ in ann.get("assigned_occupations", []) if occ.get('id')}
                        if not labels:
                            category = "No Label Given"
                        else:
                            category = frozenset(labels)
                    
                    if category in category_to_idx:
                        col_idx = category_to_idx[category]
                        ratings_matrix[row_idx, col_idx] += 1

            kappa = fleiss_kappa(ratings_matrix, method='fleiss')
            
            interp = "Poor"
            if kappa > 0.8: interp = "Almost Perfect"
            elif kappa > 0.6: interp = "Substantial"
            elif kappa > 0.4: interp = "Moderate"
            elif kappa > 0.2: interp = "Fair"
            elif kappa > 0.0: interp = "Slight"
            
            print(f"    - Fleiss' Kappa: {kappa:.4f}")
            print(f"    - Interpretation: {interp} agreement")

        except ImportError:
            print("    Skipping Fleiss' Kappa calculation: `statsmodels` is not installed.")
            print("    Please run `pip install statsmodels`")
        except Exception as e:
            print(f"    Could not calculate Fleiss' Kappa due to an error: {e}")

    # --- 5. Hierarchical Agreement Analysis ---
    print("\n\nHierarchical Agreement Analysis (by level):")
    
    try:
        from sklearn.metrics import cohen_kappa_score
        h_kappa_available = True
    except ImportError:
        h_kappa_available = False

    def get_ancestor(label_id: str, level: int) -> Optional[str]:
        """
        Gets the ancestor of a hierarchical label ID at a specific level.
        e.g., get_ancestor("L1.2.3.4", 1) -> "L1.2.3"
        Level is 0-indexed depth.
        """
        if not isinstance(label_id, str):
            return label_id
        parts = label_id.split('.')
        # Number of path segments determines depth. "L1.2" has 1 segment, depth 0.
        num_path_parts = len(parts) - 1
        if level < 0: return None
        if level >= num_path_parts:
            return label_id # At or beyond its own depth, it's its own ancestor
        
        # level 0 -> take prefix + 1st path part -> parts[:2]
        ancestor_parts = parts[:level + 2]
        return ".".join(ancestor_parts)

    hierarchical_taxonomies = ['esco', 'tree']
    for tax_type in hierarchical_taxonomies:
        if tax_type not in annotations_by_tax:
            continue
        print(f"\n--- Taxonomy: {tax_type.upper()} ---")

        # Determine max depth for this taxonomy from the data
        max_depth = 0
        all_labels_in_tax = set()
        level_label_counts = Counter()

        id_map_for_tax = id_to_title_maps.get(tax_type, {})
        for label_id in id_map_for_tax.keys():
            depth = len(label_id.split('.')) - 2 # L0.1 is depth 0
            if depth >= 0:
                 level_label_counts[depth] += 1
            if depth > max_depth:
                max_depth = depth

        for llm_data in annotations_by_tax[tax_type].values():
            for annotation in llm_data.values():
                 if not annotation.get("is_other"):
                    for occ in annotation.get("assigned_occupations", []):
                        if occ.get('id'):
                            all_labels_in_tax.add(occ['id'])
        
        if not all_labels_in_tax:
            print("  No labeled data to analyze for hierarchical agreement.")
            continue
        
        # Recalculate max_depth based on actual observed labels, which might be shallower than full taxonomy
        observed_max_depth = 0
        for label_id in all_labels_in_tax:
            depth = len(label_id.split('.')) - 2
            if depth > observed_max_depth:
                observed_max_depth = depth
        
        print(f"  (Analyzing from Level 0 to max observed depth {observed_max_depth})")

        llms_for_tax = sorted(list(annotations_by_tax[tax_type].keys()))
        if len(llms_for_tax) < 2:
            print("  Not enough LLMs for hierarchical comparison.")
            continue

        if not h_kappa_available:
            print("    Skipping Cohen's Kappa calculation for levels: `scikit-learn` is not installed.")

        for level in range(observed_max_depth + 1):
            num_labels_at_level = level_label_counts.get(level, 0)
            print(f"\n  --- Analysis at Level {level} (Labels in taxonomy at this level: {num_labels_at_level}) ---")
            
            # Pairwise agreement at this level
            for llm1, llm2 in combinations(llms_for_tax, 2):
                agreement_count = 0
                
                ann1_map = annotations_by_tax[tax_type][llm1]
                ann2_map = annotations_by_tax[tax_type][llm2]
                common_job_ids = set(ann1_map.keys()) & set(ann2_map.keys())
                comparison_count = len(common_job_ids)

                ancestors1_for_kappa = []
                ancestors2_for_kappa = []

                for job_id in common_job_ids:
                    ann1 = ann1_map[job_id]
                    ann2 = ann2_map[job_id]

                    is_other1 = ann1.get("is_other", False)
                    ancestors1 = {get_ancestor(occ['id'], level) for occ in ann1.get("assigned_occupations", []) if occ.get('id')}
                    cat1 = "Other" if is_other1 else (frozenset(ancestors1) if ancestors1 else "No Label Given")
                    ancestors1_for_kappa.append(str(cat1))

                    is_other2 = ann2.get("is_other", False)
                    ancestors2 = {get_ancestor(occ['id'], level) for occ in ann2.get("assigned_occupations", []) if occ.get('id')}
                    cat2 = "Other" if is_other2 else (frozenset(ancestors2) if ancestors2 else "No Label Given")
                    ancestors2_for_kappa.append(str(cat2))

                    # Agreement only if NEITHER is 'Other'
                    if not ann1.get("is_other", False) and not ann2.get("is_other", False):
                        # ... and their non-empty ancestor sets match.
                        if ancestors1 and ancestors1 == ancestors2:
                            agreement_count += 1
                
                if comparison_count > 0:
                    agreement_ratio = agreement_count / comparison_count
                    print(f"    - Pairwise {llm1:>{max_llm_name_len}} vs {llm2:>{max_llm_name_len}}: {agreement_ratio:.2%} agreement ({agreement_count}/{comparison_count})")
                    
                    if h_kappa_available and len(ancestors1_for_kappa) > 0:
                        try:
                            kappa = cohen_kappa_score(ancestors1_for_kappa, ancestors2_for_kappa)
                            interp = "Poor"
                            if kappa > 0.8: interp = "Almost Perfect"
                            elif kappa > 0.6: interp = "Substantial"
                            elif kappa > 0.4: interp = "Moderate"
                            elif kappa > 0.2: interp = "Fair"
                            elif kappa > 0.0: interp = "Slight"
                            print(f"      - Cohen's Kappa: {kappa:.4f} ({interp} agreement)")
                        except Exception as e:
                            print(f"      - Could not calculate Cohen's Kappa for level {level}: {e}")
                else:
                     print(f"    - Pairwise {llm1:>{max_llm_name_len}} vs {llm2:>{max_llm_name_len}}: No comparable non-Other annotations.")
            
            # N-way agreement at this level
            if len(llms_for_tax) > 2:
                n_way_agreement_count = 0
                
                common_job_ids_all = set.intersection(*[set(annotations_by_tax[tax_type][llm].keys()) for llm in llms_for_tax])
                n_way_comparison_count = len(common_job_ids_all)

                for job_id in common_job_ids_all:
                    all_anns = [annotations_by_tax[tax_type][llm][job_id] for llm in llms_for_tax]
                    
                    # If any model said 'Other', it's a disagreement
                    if any(ann.get("is_other", False) for ann in all_anns):
                        continue

                    first_ann = all_anns[0]
                    ancestors_ref = {get_ancestor(occ['id'], level) for occ in first_ann.get("assigned_occupations", []) if occ.get('id')}

                    # If the first set of ancestors is empty, it's a disagreement
                    if not ancestors_ref:
                        continue

                    all_agree = True
                    for i in range(1, len(all_anns)):
                        next_ann = all_anns[i]
                        next_ancestors = {get_ancestor(occ['id'], level) for occ in next_ann.get("assigned_occupations", []) if occ.get('id')}
                        
                        if ancestors_ref != next_ancestors:
                            all_agree = False
                            break
                    if all_agree:
                        n_way_agreement_count += 1
                
                if n_way_comparison_count > 0:
                    n_way_ratio = n_way_agreement_count / n_way_comparison_count
                    print(f"    - Overall agreement for all {len(llms_for_tax)} LLMs: {n_way_ratio:.2%} ({n_way_agreement_count}/{n_way_comparison_count})")

            # Fleiss' Kappa at this level
            if len(llms_for_tax) > 1:
                try:
                    common_job_ids_level = sorted(list(set.intersection(*[set(annotations_by_tax[tax_type][llm].keys()) for llm in llms_for_tax])))
                    
                    if not common_job_ids_level:
                        continue

                    # Determine categories for this level
                    level_categories = set()
                    for job_id in common_job_ids_level:
                        for llm in llms_for_tax:
                            ann = annotations_by_tax[tax_type][llm][job_id]
                            if ann.get("is_other", False):
                                category = "Other"
                            else:
                                ancestors = {get_ancestor(occ['id'], level) for occ in ann.get("assigned_occupations", []) if occ.get('id')}
                                if not ancestors:
                                    category = "No Label Given"
                                else:
                                    category = frozenset(ancestors)
                            level_categories.add(category)

                    category_list_level = sorted(list(level_categories), key=lambda x: str(x))
                    category_to_idx_level = {cat: i for i, cat in enumerate(category_list_level)}

                    num_subjects_level = len(common_job_ids_level)
                    num_categories_level = len(category_list_level)
                    ratings_matrix_level = np.zeros((num_subjects_level, num_categories_level), dtype=int)
                    job_id_to_idx_level = {job_id: i for i, job_id in enumerate(common_job_ids_level)}

                    for llm in llms_for_tax:
                        for job_id in common_job_ids_level:
                            row_idx = job_id_to_idx_level[job_id]
                            ann = annotations_by_tax[tax_type][llm][job_id]

                            if ann.get("is_other", False):
                                category = "Other"
                            else:
                                ancestors = {get_ancestor(occ['id'], level) for occ in ann.get("assigned_occupations", []) if occ.get('id')}
                                if not ancestors:
                                    category = "No Label Given"
                                else:
                                    category = frozenset(ancestors)
                            
                            if category in category_to_idx_level:
                                col_idx = category_to_idx_level[category]
                                ratings_matrix_level[row_idx, col_idx] += 1
                    
                    if np.sum(ratings_matrix_level) > 0:
                        kappa_level = fleiss_kappa(ratings_matrix_level, method='fleiss')
                        interp = "Poor"
                        if kappa_level > 0.8: interp = "Almost Perfect"
                        elif kappa_level > 0.6: interp = "Substantial"
                        elif kappa_level > 0.4: interp = "Moderate"
                        elif kappa_level > 0.2: interp = "Fair"
                        elif kappa_level > 0.0: interp = "Slight"
                        print(f"    - Fleiss' Kappa ({len(llms_for_tax)} LLMs): {kappa_level:.4f} ({interp} agreement)")
                    
                except Exception as e:
                    print(f"    - Could not calculate Fleiss' Kappa for Level {level} due to an error: {e}")

    
    # --- 6. Hallucination / Mismatch Analysis ---
    if id_to_title_maps:
        print("\n\nHallucination/Mismatch Analysis (ID vs. Title):")
        mismatch_cases = []

        for tax_type in sorted(annotations_by_tax.keys()):
            print(f"\n--- Taxonomy: {tax_type.upper()} ---")
            
            id_map = id_to_title_maps.get(tax_type)
            if not id_map:
                print("  ID-to-Title map not available. Skipping analysis for this taxonomy.")
                continue

            for llm in sorted(annotations_by_tax[tax_type].keys()):
                mismatch_count = 0
                total_labels_checked = 0

                for job_id, annotation in annotations_by_tax[tax_type][llm].items():
                    if not annotation.get("is_other", False):
                        assigned_occupations = annotation.get("assigned_occupations", [])
                        total_labels_checked += len(assigned_occupations)

                        for assigned_occ in assigned_occupations:
                            llm_id = assigned_occ.get('id')
                            llm_title = assigned_occ.get('title')

                            if not all([llm_id, llm_title]):
                                continue # Skip if ID or title is missing in annotation

                            expected_title = id_map.get(llm_id)
                            
                            # A mismatch occurs if the ID is not in our map OR if the title doesn't match
                            if expected_title is None or expected_title != llm_title:
                                mismatch_count += 1
                                mismatch_cases.append({
                                    "job_id": job_id,
                                    "llm": llm,
                                    "taxonomy_type": tax_type,
                                    "mismatched_id": llm_id,
                                    "llm_title": llm_title,
                                    "expected_title": expected_title if expected_title else "ID NOT FOUND"
                                })

                if total_labels_checked > 0:
                    mismatch_ratio = mismatch_count / total_labels_checked
                    print(f"  - {llm:>{max_llm_name_len}}: {mismatch_ratio:.2%} mismatch rate ({mismatch_count}/{total_labels_checked} labels)")
                else:
                    print(f"  - {llm:>{max_llm_name_len}}: No labeled annotations to check.")
        
        # Save mismatch cases to a CSV file
        if mismatch_cases:
            mismatch_df = pd.DataFrame(mismatch_cases)
            mismatch_csv_path = bulk_output_dir.parent / f"{bulk_output_dir.name}_annotation_mismatches.csv"
            try:
                mismatch_df.to_csv(mismatch_csv_path, index=False, encoding='utf-8')
                print(f"\nSaved {len(mismatch_cases)} mismatch cases to: {mismatch_csv_path}")
            except Exception as e:
                print(f"Error saving mismatches DataFrame to CSV: {e}")

    # --- Ground Truth Analysis Section ---
    if ground_truth_df is not None:
        gt_scenarios = []
        if 'canonical_label' in ground_truth_df.columns:
            gt_scenarios.append({
                "name": "canonical_label",
                "series": ground_truth_df['canonical_label']
            })
        if 'canonical_label_taxonomy' in ground_truth_df.columns:
            gt_scenarios.append({
                "name": "canonical_label_taxonomy",
                "series": ground_truth_df['canonical_label_taxonomy']
            })
        if 'canonical_label' in ground_truth_df.columns and 'canonical_label_taxonomy' in ground_truth_df.columns:
            # Create a series where each value is a set of the two labels
            union_series = ground_truth_df.apply(
                lambda row: {str(row['canonical_label']), str(row['canonical_label_taxonomy'])},
                axis=1
            )
            gt_scenarios.append({
                "name": "union_of_labels",
                "series": union_series
            })

        for scenario in gt_scenarios:
            _perform_ground_truth_analysis(
                scenario_name=scenario['name'],
                ground_truth_series=scenario['series'],
                annotations_by_tax=annotations_by_tax,
                id_to_title_maps=id_to_title_maps,
                bulk_output_dir=bulk_output_dir,
                max_llm_name_len=max_llm_name_len,
                ground_truth_df=ground_truth_df # Pass the full df for context
            )


    print("\n--- End of Annotation Evaluation ---")


def _perform_ground_truth_analysis(
    scenario_name: str,
    ground_truth_series: pd.Series,
    annotations_by_tax: Dict,
    id_to_title_maps: Dict,
    bulk_output_dir: Path,
    max_llm_name_len: int,
    ground_truth_df: pd.DataFrame
):
    """
    Helper function to run all ground-truth-based evaluations for a given GT series.
    A ground_truth_series can contain single string labels or sets of string labels.
    """
    print(f"\n\n{'='*20}")
    print(f"Ground Truth Agreement Analysis (Scenario: {scenario_name})")
    print(f"{'='*20}")

    gt_dict = ground_truth_series.to_dict()
    is_union_scenario = isinstance(ground_truth_series.iloc[0], set)

    # --- Create and save a comparison DataFrame ---
    comparison_df = ground_truth_df.copy()
    if comparison_df.index.name == 'job_id':
         comparison_df.reset_index(inplace=True)
    
    comparison_df['job_id'] = comparison_df['job_id'].astype(str)
    
    tax_type_for_comparison = 'tree'
    if tax_type_for_comparison in annotations_by_tax:
        llms_for_tax = sorted(list(annotations_by_tax[tax_type_for_comparison].keys()))
        for llm in llms_for_tax:
            llm_annotations = {}
            for job_id, annotation in annotations_by_tax[tax_type_for_comparison][llm].items():
                if annotation.get("is_other", False):
                    llm_annotations[job_id] = "Other"
                else:
                    assigned_titles = {str(occ.get('title')) for occ in annotation.get("assigned_occupations", []) if occ.get('title')}
                    llm_annotations[job_id] = "|".join(sorted(list(assigned_titles))) if assigned_titles else ""
            
            comparison_df[llm] = comparison_df['job_id'].map(llm_annotations)
    
    comparison_pickle_path = bulk_output_dir.parent / f"{bulk_output_dir.name}_ground_truth_comparison_{scenario_name}.pkl"
    try:
        comparison_df.to_pickle(comparison_pickle_path)
        print(f"\nSaved ground truth comparison results to: {comparison_pickle_path}")
    except Exception as e:
        print(f"Error saving ground truth comparison DataFrame to pickle: {e}")

    individual_disagreements = []
    majority_vote_disagreements = []

    for tax_type in ['tree']: # Focus on tree taxonomy for GT comparison
        print(f"\n--- Taxonomy: {tax_type.upper()} ---")

        # --- Individual LLM Agreement with Ground Truth ---
        print("  --- Individual LLM vs. Ground Truth ---")
        for llm in sorted(annotations_by_tax[tax_type].keys()):
            agreement_count = 0
            comparison_count = 0
            
            for job_id, annotation in annotations_by_tax[tax_type][llm].items():
                if job_id in gt_dict:
                    comparison_count += 1
                    gt_labels = gt_dict[job_id]
                    if not is_union_scenario: gt_labels = {str(gt_labels)}

                    is_other = annotation.get("is_other", False)
                    assigned_labels = {str(occ.get('title')) for occ in annotation.get("assigned_occupations", []) if occ.get('title')}

                    agreed = False
                    is_gt_other = any(l.lower() == 'other' for l in gt_labels)
                    if not is_other and not is_gt_other:
                        if not assigned_labels.isdisjoint(gt_labels):
                            agreed = True

                    if agreed:
                        agreement_count += 1
                    else:
                        llm_annotation_str = "Other" if is_other else "|".join(sorted(list(assigned_labels)))
                        individual_disagreements.append({
                            'job_id': job_id,
                            'ground_truth_label': "|".join(sorted(list(gt_labels))),
                            'llm': llm,
                            'llm_annotation': llm_annotation_str
                        })
            
            if comparison_count > 0:
                agreement_ratio = agreement_count / comparison_count
                print(f"    - {llm:>{max_llm_name_len}}: {agreement_ratio:.2%} agreement ({agreement_count}/{comparison_count} agreements)")
            else:
                print(f"    - {llm:>{max_llm_name_len}}: No overlapping jobs with ground truth.")

        # --- Any LLM Agreement with Ground Truth ---
        print("\n  --- Any LLM vs. Ground Truth ---")
        llms_in_tax = list(annotations_by_tax[tax_type].keys())
        if not llms_in_tax:
            print("    No LLM annotations available for this taxonomy to compare with ground truth.")
        else:
            any_llm_agreement_count = 0
            comparison_count = 0

            for job_id, gt_labels in gt_dict.items():
                if any(job_id in annotations_by_tax[tax_type].get(llm, {}) for llm in llms_in_tax):
                    comparison_count += 1
                    if not is_union_scenario: gt_labels = {str(gt_labels)}
                    
                    any_llm_agreed = False
                    is_gt_other = any(l.lower() == 'other' for l in gt_labels)
                    
                    for llm in llms_in_tax:
                        if job_id not in annotations_by_tax[tax_type].get(llm, {}):
                            continue

                        annotation = annotations_by_tax[tax_type][llm][job_id]
                        is_other = annotation.get("is_other", False)
                        assigned_labels = {str(occ.get('title')) for occ in annotation.get("assigned_occupations", []) if occ.get('title')}

                        if not is_other and not is_gt_other:
                            if not assigned_labels.isdisjoint(gt_labels):
                                any_llm_agreed = True
                                break
                    
                    if any_llm_agreed:
                        any_llm_agreement_count += 1
            
            if comparison_count > 0:
                agreement_ratio = any_llm_agreement_count / comparison_count
                print(f"    - At Least One Match Rate: {agreement_ratio:.2%} ({any_llm_agreement_count}/{comparison_count} jobs where at least one LLM's annotation matched any ground truth label)")
            else:
                print("    - No overlapping jobs found between any LLM and ground truth.")

        # --- Majority Vote Agreement with Ground Truth ---
        print("\n  --- Majority Vote vs. Ground Truth ---")
        if len(llms_in_tax) < 2:
             print("    Not enough LLMs for a meaningful majority vote.")
        else:
            agreement_count = 0
            comparison_count = 0
            tied_votes = 0

            for job_id, gt_labels in gt_dict.items():
                votes = Counter()
                voters = [llm for llm in llms_in_tax if job_id in annotations_by_tax[tax_type][llm]]
                if not voters: continue

                comparison_count += 1
                for llm in voters:
                    annotation = annotations_by_tax[tax_type][llm][job_id]
                    is_other = annotation.get("is_other", False)
                    assigned_labels = tuple(sorted({str(occ.get('title')) for occ in annotation.get("assigned_occupations", []) if occ.get('title')}))
                    
                    if is_other: votes['__OTHER__'] += 1
                    elif assigned_labels: votes[assigned_labels] += 1
                    else: votes['__EMPTY__'] += 1
                
                if not votes: continue

                winners = votes.most_common(2)
                if len(winners) == 1 or (len(winners) > 1 and winners[0][1] > winners[1][1]):
                    majority_vote = winners[0][0]
                    if not is_union_scenario: gt_labels = {str(gt_labels)}
                    is_gt_other = any(l.lower() == 'other' for l in gt_labels)

                    agreed = False
                    if majority_vote == '__OTHER__' and is_gt_other:
                        agreed = True
                    elif isinstance(majority_vote, tuple) and not is_gt_other:
                         if not set(majority_vote).isdisjoint(gt_labels):
                              agreed = True

                    if agreed:
                        agreement_count += 1
                    else:
                        majority_vote_str = "Other" if majority_vote == '__OTHER__' else "|".join(majority_vote) if isinstance(majority_vote, tuple) else "EMPTY"
                        majority_vote_disagreements.append({
                            'job_id': job_id,
                            'ground_truth_label': "|".join(sorted(list(gt_labels))),
                            'majority_vote_annotation': majority_vote_str,
                        })
                else:
                    tied_votes += 1
            
            effective_comparisons = comparison_count - tied_votes
            if effective_comparisons > 0:
                agreement_ratio = agreement_count / effective_comparisons
                print(f"    - Majority Vote: {agreement_ratio:.2%} agreement ({agreement_count}/{effective_comparisons} decided votes)")
                if tied_votes > 0:
                    print(f"      (Excluded {tied_votes} jobs due to tied votes)")
            else:
                print(f"    - No clear majority decisions were found across {comparison_count} comparable jobs.")

    if individual_disagreements:
        disagreements_df = pd.DataFrame(individual_disagreements)
        disagreements_csv_path = bulk_output_dir.parent / f"{bulk_output_dir.name}_individual_disagreements_{scenario_name}.csv"
        disagreements_df.to_csv(disagreements_csv_path, index=False, encoding='utf-8')
        print(f"\nSaved {len(individual_disagreements)} individual disagreement cases to: {disagreements_csv_path}")

    if majority_vote_disagreements:
        majority_df = pd.DataFrame(majority_vote_disagreements)
        majority_csv_path = bulk_output_dir.parent / f"{bulk_output_dir.name}_majority_vote_disagreements_{scenario_name}.csv"
        majority_df.to_csv(majority_csv_path, index=False, encoding='utf-8')
        print(f"\nSaved {len(majority_vote_disagreements)} majority vote disagreement cases to: {majority_csv_path}")

    if id_to_title_maps.get('tree'):
        print("\n\nGround Truth Agreement Analysis (Level-by-Level for 'tree' taxonomy)")
        tax_type = 'tree'
        tree_id_to_title = id_to_title_maps[tax_type]
        tree_title_to_id = {v: k for k, v in tree_id_to_title.items()}

        max_depth, level_label_counts = 0, Counter()
        for label_id in tree_id_to_title.keys():
            depth = len(label_id.split('.')) - 2
            if depth >= 0: level_label_counts[depth] += 1
            if depth > max_depth: max_depth = depth

        for level in range(max_depth + 1):
            print(f"\n  --- Analysis at Level {level} (Labels in taxonomy at this level: {level_label_counts.get(level, 0)}) ---")
            
            for llm in sorted(annotations_by_tax[tax_type].keys()):
                agreement_count, comparison_count = 0, 0
                for job_id, llm_ann in annotations_by_tax[tax_type][llm].items():
                    if job_id in gt_dict:
                        comparison_count += 1
                        gt_labels = gt_dict[job_id]
                        if not is_union_scenario: gt_labels = {str(gt_labels)}

                        is_llm_other = llm_ann.get("is_other", False)
                        is_gt_other = any(l.lower() == 'other' for l in gt_labels)

                        if not is_llm_other and not is_gt_other:
                            gt_ids = {tree_title_to_id.get(l) for l in gt_labels if tree_title_to_id.get(l)}
                            gt_ids.discard(None) # Remove None if a label wasn't found

                            # Check for missing labels and print a warning
                            if len(gt_ids) != len(gt_labels):
                                missing_labels = [l for l in gt_labels if tree_title_to_id.get(l) is None]
                                print(f"Warning: For job_id '{job_id}', the following ground truth labels were not found in the taxonomy map and will be ignored: {missing_labels}")
                            
                            if not gt_ids:
                                continue
                            
                            gt_ancestor_ids = {get_ancestor(gid, level) for gid in gt_ids if get_ancestor(gid, level)}
                            if not gt_ancestor_ids: continue

                            llm_ancestor_ids = {get_ancestor(occ.get('id'), level) for occ in llm_ann.get("assigned_occupations", []) if occ.get('id')}
                            llm_ancestor_ids.discard(None)
                            
                            if not llm_ancestor_ids.isdisjoint(gt_ancestor_ids):
                                agreement_count += 1
                
                if comparison_count > 0:
                    ratio = agreement_count / comparison_count
                    print(f"    - {llm:>{max_llm_name_len}}: {ratio:.2%} agreement ({agreement_count}/{comparison_count})")
    
    # --- Conformity Ranking, Proximity, and Relaxed Agreement ---
    tax_type = 'tree'
    if tax_type not in annotations_by_tax or tax_type not in id_to_title_maps:
        return # Cannot perform these analyses without a tree taxonomy

    tree_title_to_id = {v: k for k, v in id_to_title_maps[tax_type].items()}

    # --- Conformity Ranking Analysis (Leave-One-Out Majority Vote) ---
    if is_union_scenario:
        print("\n\nSkipping Conformity Ranking for 'union' scenario as it requires single-label ground truth.")
    else:
        print("\n\nConformity Ranking Analysis (Leave-One-Out vs. Majority Vote):")
        print("      (This metric ranks each annotator by its agreement with the majority vote of all OTHER annotators.)")
        gt_as_annotator = {}
        for job_id, gt_title in gt_dict.items():
            gt_id = tree_title_to_id.get(str(gt_title))
            assigned_occupations = [{"id": gt_id, "title": str(gt_title)}] if gt_id else []
            gt_as_annotator[str(job_id)] = {
                "assigned_occupations": assigned_occupations, "is_other": not gt_id
            }

        all_annotators = {'ground_truth': gt_as_annotator, **annotations_by_tax[tax_type]}
        annotator_names = sorted(list(all_annotators.keys()))
        
        max_depth = max((len(lid.split('.')) - 2 for lid in id_to_title_maps[tax_type].keys()), default=-1)
                
        for level in range(max_depth + 1):
            print(f"\n  --- Conformity Analysis at Level {level} ---")
            conformity_scores = {}
            for target_name in annotator_names:
                # ... (rest of conformity logic)
                target_annotations = all_annotators[target_name]
                other_names = [name for name in annotator_names if name != target_name]
                
                agreement_count, comparison_count, tied_votes = 0, 0, 0

                for job_id, target_ann in target_annotations.items():
                    votes = Counter()
                    voters = [name for name in other_names if job_id in all_annotators.get(name, {})]
                    if len(voters) < 2: continue
                    comparison_count += 1
                    
                    for voter_name in voters:
                        voter_ann = all_annotators[voter_name][job_id]
                        if voter_ann.get("is_other", False):
                            votes['__OTHER__'] += 1
                        else:
                            ancestor_ids = {get_ancestor(occ.get('id'), level) for occ in voter_ann.get("assigned_occupations", []) if occ.get('id')}
                            ancestor_ids.discard(None)
                            if ancestor_ids: votes[frozenset(ancestor_ids)] += 1
                            else: votes['__EMPTY__'] += 1

                    if not votes: continue

                    winners = votes.most_common(2)
                    if len(winners) > 1 and winners[0][1] == winners[1][1]:
                        tied_votes += 1
                        continue
                    
                    majority_vote_winner = winners[0][0]
                    target_vote = None
                    if target_ann.get("is_other", False):
                        target_vote = '__OTHER__'
                    else:
                        target_ancestors = {get_ancestor(occ.get('id'), level) for occ in target_ann.get("assigned_occupations", []) if occ.get('id')}
                        target_ancestors.discard(None)
                        if target_ancestors: target_vote = frozenset(target_ancestors)
                        else: target_vote = '__EMPTY__'
                    
                    if target_vote == majority_vote_winner:
                        agreement_count += 1

                effective_comparisons = comparison_count - tied_votes
                if effective_comparisons > 0:
                    conformity_scores[target_name] = agreement_count / effective_comparisons
                else:
                    conformity_scores[target_name] = 0.0

            if conformity_scores:
                print(f"    Ranking of annotators by conformity score:")
                sorted_scores = sorted(conformity_scores.items(), key=lambda item: item[1], reverse=True)
                for i, (name, score) in enumerate(sorted_scores):
                    print(f"      {i+1}. {name:>{max_llm_name_len}}: {score:.2%}")
            else:
                print(f"    Could not compute conformity scores at this level.")
    
    # --- Hierarchical Proximity Score Analysis (vs. Ground Truth) ---
    print("\n\nHierarchical Proximity Score Analysis (Distance to Ground Truth):")
    print("      (Lower score is better. Measures average steps in tree between prediction and ground truth.)")
    OTHER_PENALTY = 10 
    
    for llm in sorted(annotations_by_tax[tax_type].keys()):
        distances = []
        for job_id, llm_ann in annotations_by_tax[tax_type][llm].items():
            if job_id in gt_dict:
                gt_labels = gt_dict[job_id]
                if not is_union_scenario: gt_labels = {str(gt_labels)}
                
                is_llm_other = llm_ann.get("is_other", False)
                is_gt_other = any(l.lower() == 'other' for l in gt_labels)

                distance = OTHER_PENALTY
                if is_llm_other and is_gt_other:
                    distance = 0
                elif not is_llm_other and not is_gt_other:
                    llm_occupations = llm_ann.get("assigned_occupations", [])
                    if llm_occupations:
                        llm_id = llm_occupations[0].get('id')
                        gt_ids = {tree_title_to_id.get(l) for l in gt_labels if l in tree_title_to_id}
                        
                        if llm_id and gt_ids:
                            dists_to_gts = [calculate_hierarchical_distance(g_id, llm_id) for g_id in gt_ids]
                            valid_dists = [d for d in dists_to_gts if d is not None]
                            if valid_dists:
                                distance = min(valid_dists)
                distances.append(distance)
        
        if distances:
            avg_distance = sum(distances) / len(distances)
            print(f"    - {llm:>{max_llm_name_len}}: {avg_distance:.2f} average distance")

    # --- Relaxed Agreement Score (Distance <= 2) vs. Ground Truth ---
    print("\n\nRelaxed Agreement Score (Distance <= 2) vs. Ground Truth:")
    print("      (Counts an annotation as 'correct' if its hierarchical distance from the ground truth is 2 or less.)")

    print("\n  --- Individual LLM Relaxed Agreement ---")
    ratios = []
    for llm in sorted(annotations_by_tax[tax_type].keys()):
        agreement_count, comparison_count = 0, 0
        for job_id, llm_ann in annotations_by_tax[tax_type][llm].items():
            if job_id in gt_dict:
                comparison_count += 1
                gt_labels = gt_dict[job_id]
                if not is_union_scenario: gt_labels = {str(gt_labels)}

                agreed = False
                is_llm_other = llm_ann.get("is_other", False)
                is_gt_other = any(l.lower() == 'other' for l in gt_labels)

                if is_llm_other and is_gt_other:
                    agreed = True
                elif not is_llm_other and not is_gt_other:
                    llm_id = llm_ann.get("assigned_occupations", [{}])[0].get('id')
                    gt_ids = {tree_title_to_id.get(l) for l in gt_labels if l in tree_title_to_id}
                    if llm_id and gt_ids:
                        for g_id in gt_ids:
                            distance = calculate_hierarchical_distance(g_id, llm_id)
                            if distance is not None and distance <= 2:
                                agreed = True
                                break
                if agreed:
                    agreement_count += 1
        
        if comparison_count > 0:
            ratio = agreement_count / comparison_count
            print(f"    - {llm:>{max_llm_name_len}}: {ratio:.2%} agreement ({agreement_count}/{comparison_count} jobs)")
            ratios.append(ratio)
    
    if ratios:
        print(f"    - Average ratio: {sum(ratios) / len(ratios):.2%}")

    # --- Group Relaxed Agreement ---
    print("\n  --- Group Relaxed Agreement ---")
    llms_in_tax = list(annotations_by_tax[tax_type].keys())
    if not llms_in_tax:
        print("    No LLM annotations available for this taxonomy to compare.")
    else:
        # Any LLM Relaxed Agreement
        any_llm_agreement_count = 0
        any_comparison_count = 0
        for job_id, gt_labels in gt_dict.items():
            if any(job_id in annotations_by_tax[tax_type].get(llm, {}) for llm in llms_in_tax):
                any_comparison_count += 1
                if not is_union_scenario: gt_labels = {str(gt_labels)}
                
                any_llm_agreed = False
                is_gt_other = any(l.lower() == 'other' for l in gt_labels)

                for llm in llms_in_tax:
                    if job_id not in annotations_by_tax[tax_type].get(llm, {}): continue
                    
                    annotation = annotations_by_tax[tax_type][llm][job_id]
                    is_llm_other = annotation.get("is_other", False)
                    
                    agreed = False
                    if is_llm_other and is_gt_other:
                        agreed = True
                    elif not is_llm_other and not is_gt_other:
                        llm_id = annotation.get("assigned_occupations", [{}])[0].get('id')
                        gt_ids = {tree_title_to_id.get(l) for l in gt_labels if l in tree_title_to_id}
                        if llm_id and gt_ids:
                            for g_id in gt_ids:
                                distance = calculate_hierarchical_distance(g_id, llm_id)
                                if distance is not None and distance <= 2:
                                    agreed = True
                                    break
                    if agreed:
                        any_llm_agreed = True
                        break
                
                if any_llm_agreed:
                    any_llm_agreement_count += 1
        
        if any_comparison_count > 0:
            agreement_ratio = any_llm_agreement_count / any_comparison_count
            print(f"    - Any LLM Agreement: {agreement_ratio:.2%} ({any_llm_agreement_count}/{any_comparison_count})")

        # Majority Vote Relaxed Agreement
        if len(llms_in_tax) >= 2:
            maj_agreement_count = 0
            maj_comparison_count = 0
            maj_tied_votes = 0

            for job_id, gt_labels in gt_dict.items():
                votes = Counter()
                voters = [llm for llm in llms_in_tax if job_id in annotations_by_tax[tax_type][llm]]
                if not voters: continue

                maj_comparison_count += 1
                for llm in voters:
                    annotation = annotations_by_tax[tax_type][llm][job_id]
                    is_other = annotation.get("is_other", False)
                    assigned_labels = tuple(sorted({str(occ.get('title')) for occ in annotation.get("assigned_occupations", []) if occ.get('title')}))
                    
                    if is_other: votes['__OTHER__'] += 1
                    elif assigned_labels: votes[assigned_labels] += 1
                    else: votes['__EMPTY__'] += 1
                
                if not votes: continue

                winners = votes.most_common(2)
                if len(winners) > 1 and winners[0][1] == winners[1][1]:
                    maj_tied_votes += 1
                    continue
                
                majority_vote = winners[0][0]
                if not is_union_scenario: gt_labels = {str(gt_labels)}
                is_gt_other = any(l.lower() == 'other' for l in gt_labels)

                agreed = False
                if majority_vote == '__OTHER__' and is_gt_other:
                    agreed = True
                elif isinstance(majority_vote, tuple) and not is_gt_other:
                    majority_ids = {tree_title_to_id.get(l) for l in majority_vote if l in tree_title_to_id}
                    gt_ids = {tree_title_to_id.get(l) for l in gt_labels if l in tree_title_to_id}
                    
                    if majority_ids and gt_ids:
                        for m_id in majority_ids:
                            for g_id in gt_ids:
                                distance = calculate_hierarchical_distance(g_id, m_id)
                                if distance is not None and distance <= 2:
                                    agreed = True
                                    break
                            if agreed: break
                
                if agreed:
                    maj_agreement_count += 1

            effective_comparisons = maj_comparison_count - maj_tied_votes
            if effective_comparisons > 0:
                agreement_ratio = maj_agreement_count / effective_comparisons
                print(f"    - Majority Vote Agreement: {agreement_ratio:.2%} ({maj_agreement_count}/{effective_comparisons})")


def create_correct_esco_title_map(data_dir: Path):
    """
    Creates the correct ESCO ID-to-Title map if it's missing or incorrect.
    This is a utility to fix the state without re-running all annotations.
    """
    id_to_code_map_path = data_dir / "esco_id_to_code_map.json"
    id_to_title_map_path = data_dir / "esco_id_to_title_map.json"

    # For this to work, we need the map of LLM IDs to ESCO codes.
    if not id_to_code_map_path.exists():
        print(f"Info: Cannot create ESCO title map because '{id_to_code_map_path.name}' is missing. Please run the annotation formatting once.")
        return

    # Check if the title map already seems correct to avoid extra work.
    if id_to_title_map_path.exists():
        try:
            with open(id_to_title_map_path, 'r', encoding='utf-8') as f:
                sample_map = json.load(f)
            # Simple heuristic: if a value contains a space, it's likely a title, not a code.
            if sample_map and any(" " in v for v in sample_map.values()):
                print("Info: Correct ESCO ID-to-Title map already exists.")
                return 
        except Exception:
             # If file is invalid, proceed to recreate it.
            pass
    
    print("Attempting to generate correct ESCO ID-to-Title map...")
    try:
        # 1. Load the existing ID -> Code map
        with open(id_to_code_map_path, 'r', encoding='utf-8') as f:
            id_to_code = json.load(f)

        # 2. Create a Code -> Title map from raw ESCO files
        code_to_title = {}
        # ISCO Groups
        with open(data_dir / "esco/en/ISCOGroups_en.csv", "r", encoding="utf-8") as f:
            reader_isco = csv.DictReader(f)
            for row in reader_isco:
                code_to_title[row['code'].strip()] = row['preferredLabel'].strip()
        # ESCO Occupations
        with open(data_dir / "esco/en/occupations_en.csv", "r", encoding="utf-8") as f:
            reader_esco = csv.DictReader(f)
            for row in reader_esco:
                code_to_title[row['code'].strip()] = row['preferredLabel'].strip()
        
        # 3. Combine them to create the ID -> Title map
        id_to_title = {llm_id: code_to_title.get(esco_code, "TITLE NOT FOUND") 
                       for llm_id, esco_code in id_to_code.items()}

        # 4. Save the new, correct map
        with open(id_to_title_map_path, 'w', encoding='utf-8') as f:
            json.dump(id_to_title, f, ensure_ascii=False, indent=2)
        print(f"Successfully created correct ESCO ID-to-Title map at: {id_to_title_map_path}")

    except FileNotFoundError as e:
        print(f"Error creating ESCO title map: Raw data file not found. {e}")
    except Exception as e:
        print(f"An unexpected error occurred while creating ESCO title map: {e}")


def load_id_to_title_maps(
    data_dir: Path, 
    tree_taxonomy_file: str,
    flat_taxonomy_file: str,
    cache_prefix: str = "",
    tree_desc_len: int = 50,
    permute_branches: bool = False,
    enabled_taxonomies: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """Loads the ID-to-Title JSON maps for all taxonomies."""
    maps = {}
    prefix = f"{cache_prefix}_" if cache_prefix else ""
    tree_taxonomy_stem = Path(tree_taxonomy_file).stem
    flat_taxonomy_stem = Path(flat_taxonomy_file).stem

    # Handle esco taxonomy (static name)
    if not enabled_taxonomies or "esco" in enabled_taxonomies:
        esco_map_path = data_dir / "esco_id_to_title_map.json"
        if esco_map_path.exists():
            try:
                with open(esco_map_path, 'r', encoding='utf-8') as f:
                    maps['esco'] = json.load(f)
            except Exception as e:
                print(f"Error loading {esco_map_path.name}: {e}")
        else:
            print(f"Warning: {esco_map_path.name} not found.")

    # Handle tree taxonomy (dynamic name)
    if not enabled_taxonomies or "tree" in enabled_taxonomies:
        desc_len_suffix = f"desc{'full' if tree_desc_len < 0 else tree_desc_len}"
        permute_suffix = "_permuted" if permute_branches else ""
        tree_map_path = data_dir / f"{prefix}{tree_taxonomy_stem}_{desc_len_suffix}{permute_suffix}_id_to_title_map.json"
        if tree_map_path.exists():
            try:
                with open(tree_map_path, 'r', encoding='utf-8') as f:
                    maps['tree'] = json.load(f)
            except Exception as e:
                print(f"Error loading {tree_map_path.name}: {e}")
        else:
            print(f"Warning: {tree_map_path.name} not found.")

    # Handle flat taxonomy (dynamic name)
    if not enabled_taxonomies or "flat" in enabled_taxonomies:
        flat_map_path = data_dir / f"{prefix}{flat_taxonomy_stem}_id_to_title_map.json"
        if flat_map_path.exists():
            try:
                with open(flat_map_path, 'r', encoding='utf-8') as f:
                    maps['flat'] = json.load(f)
            except Exception as e:
                print(f"Error loading {flat_map_path.name}: {e}")
        else:
            print(f"Warning: {flat_map_path.name} not found.")

    return maps

def call_openai_api(
    client: Any,  # Expects an initialized OpenAI client object
    prompt: str,
    model: str,   # e.g., "gpt-4o-mini"
    temperature: float,
    max_tokens: int
) -> Optional[str]:
    """
    Wrapper function to call the OpenAI Chat Completions API.
    Returns the content of the response or an error string.
    """
    if not client:
        print("OpenAI client not provided or not initialized.")
        return "API_ERROR: OpenAI client not available"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            # temperature=temperature,
            max_completion_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API (model: {model}): {e}")
        return f"API_ERROR: {e}"

def call_deepseek_api(client, prompt: str, model: str, temperature: float, max_tokens: int) -> Optional[str]:
    if not client:
        print("DeepSeek client not available.")
        return "API_ERROR: DeepSeek client not available"
    try:
        response = client.chat.completions.create( # Assuming DeepSeek uses OpenAI-compatible SDK
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling DeepSeek API ({model}): {e}")
        return f"API_ERROR: {e}"

def call_google_api(client, prompt: str, model: str, temperature: float, max_tokens: int) -> Optional[str]:
    """
    Wrapper function to call the Google Generative AI API.
    Returns the content of the response or an error string.
    """
    if not client:
        print("Google client not available.")
        return "API_ERROR: Google client not available"
    try:
        from google.genai import types
        
        if '2.5' in model:
            response = client.models.generate_content(
            model=model,
            # system_instruction="",   
            contents=[prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                # response_mime_type='application/json',
                thinking_config=types.ThinkingConfig(thinking_budget=2048) #TODO: add thinking budget demages the performance
            )
            )
        else:
            response = client.models.generate_content(
            model=model,
            # system_instruction="", 
            contents=[prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                # response_mime_type='application/json'
            )
            )
        return response.text
    except Exception as e:
        print(f"Error calling Google API ({model}): {e}")
        return f"API_ERROR: {e}"

def call_openrouter_api(client, prompt: str, model: str, temperature: float, max_tokens: int) -> Optional[str]:
    if not client:
        print("OpenRouter client not available.")
        return "API_ERROR: OpenRouter client not available"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={}  # Can be used for additional OpenRouter-specific parameters
        )
        # print(response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenRouter API ({model}): {e}")
        return f"API_ERROR: {e}"

# --- Main Execution ---
def main(
    use_column: str = 'text', 
    sample: bool = False, 
    test_data_file: str = 'indeed_test_data_1165.pkl', 
    tree_taxonomy_file: str = 'full_generated_taxonomy_with_desc.json',
    flat_taxonomy_file: str = 'tnt_clusters_full.pkl',
    cache_prefix: str = "",
    tree_desc_len: int = 50,
    bulk_output_dir: Optional[Path] = None,
    permute_branches: bool = False,
    enabled_llms: Optional[List[str]] = None,
    enabled_taxonomies: Optional[List[str]] = None
):
    """Main function to run the taxonomy evaluation script."""
    global openai_client, deepseek_client, google_client, openrouter_client # Allow main to modify global clients

    # --- Initialize LLM Clients (Example) ---
    openai_client = None
    deepseek_client = None
    google_client = None
    openrouter_client = None
    
    try:
        if os.getenv("OPENAI_API_KEY"):
            from openai import OpenAI as OpenAIClient # Renamed to avoid conflict
            openai_client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
            print("OpenAI client initialized.")
        else:
            print("OPENAI_API_KEY not found, OpenAI client not initialized.")
            openai_client = None
    except ImportError:
        print("OpenAI SDK not installed. Skipping OpenAI client initialization.")
        openai_client = None
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        openai_client = None

    try:
        if os.getenv("GOOGLE_API_KEY"):
            from google import genai
            google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            print("Google Gemini client initialized.")
        else:
            print("GOOGLE_API_KEY not found, Google client not initialized.")
    except ImportError:
        print("Google Generative AI SDK not installed. Skipping Google LLM.")
    except Exception as e:
        print(f"Failed to initialize Google client: {e}")
        google_client = None

    try:
        if os.getenv("OPENROUTER_API_KEY"):
            from openai import OpenAI as OpenRouterClient
            openrouter_client = OpenRouterClient(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY")
            )
            print("OpenRouter client initialized.")
        else:
            print("OPENROUTER_API_KEY not found, OpenRouter client not initialized.")
            openrouter_client = None
    except Exception as e:
        print(f"Failed to initialize OpenRouter client: {e}")
        openrouter_client = None

    # --- Define LLM Configurations ---
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL", "o4-mini")
    DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-preview-04-17")
    OPENROUTER_MODEL_NAME = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1-0528:free")

    llm_configurations = [
        {
            "name": "openai_default", # Internal config name
            "display_name": "openai", # For filenames and user-facing messages
            "client": openai_client,
            "api_call_function": call_openai_api, # Assumes call_openai_api is defined
            "model_name": OPENAI_MODEL_NAME,
            "temperature": 0.,
            "max_tokens": 4000,
            "enabled": openai_client is not None
        },
        {
            "name": "deepseek_default",
            "display_name": "deepseek",
            "client": deepseek_client,
            "api_call_function": call_deepseek_api, # Assumes call_deepseek_api is defined
            "model_name": DEEPSEEK_MODEL_NAME,
            "temperature": 0.,
            "max_tokens": 4000,
            "enabled": deepseek_client is not None
        },
        {
            "name": "openrouter",
            "display_name": "openrouter",
            "client": openrouter_client,
            "api_call_function": call_openrouter_api,
            "model_name": OPENROUTER_MODEL_NAME,  # or whatever model you want to use
            "temperature": 0.,
            "max_tokens": 4000,
            "enabled": openrouter_client is not None
        },
        {
            "name": "google",
            "display_name": "google", 
            "client": google_client,
            "api_call_function": call_google_api,
            "model_name": GOOGLE_MODEL_NAME,
            "temperature": 0.,
            "max_tokens": 4000,
            "enabled": google_client is not None
        },
    ]
    
    # Filter based on enabled status
    active_llm_configurations = [config for config in llm_configurations if config.get('enabled', False)]

    # Further filter if a specific list of LLMs is provided
    if enabled_llms:
        print(f"Filtering to run only specified LLMs: {enabled_llms}")
        active_llm_configurations = [
            config for config in active_llm_configurations 
            if config.get('display_name') in enabled_llms
        ]

    if not active_llm_configurations:
        print("No LLMs are actively configured or enabled. Exiting.")
        return

    # 1. Load Data
    loaded_data = load_data_files(
        test_data_path=test_data_file,
        tree_taxonomy_file=tree_taxonomy_file,
        flat_taxonomy_file=flat_taxonomy_file
    )
    if not loaded_data:
        print("Failed to load data. Exiting.")
        return

    # 2. Reformat Taxonomies
    formatted_taxonomies = process_and_reformat_all_taxonomies(
        loaded_data, 
        tree_taxonomy_file=tree_taxonomy_file,
        flat_taxonomy_file=flat_taxonomy_file,
        cache_prefix=cache_prefix,
        tree_desc_len=tree_desc_len,
        permute_branches=permute_branches,
        enabled_taxonomies=enabled_taxonomies
    )
    if not any(formatted_taxonomies.values()): 
        print("Failed to reformat taxonomies or all are empty. Exiting.")
        return
   
    # raise Exception("Stop here")
    # 3. Annotate Job Postings
    df_test = loaded_data["job_posting"] 
    if sample:
        df_test = df_test.head(10)

    if active_llm_configurations:
        if not bulk_output_dir:
            # Fallback for direct calls, though workflow usage is preferred
            test_data_stem = Path(test_data_file).stem
            bulk_output_dir = Path("output") / f"bulk_annotations_{test_data_stem}_with_{use_column}"

        run_bulk_annotations(
            df_job_postings=df_test, 
            formatted_taxonomies=formatted_taxonomies, 
            bulk_output_dir=bulk_output_dir,
            llm_configurations=active_llm_configurations,
            use_column=use_column
        )
    else:
        print("No active LLM configurations available for bulk annotation. Skipping.")

    print("\nTaxonomy Annotation Script finished.")



def eval(
    bulk_output_dir: Path,
    ground_truth_file: str = "df_test_annotated_by_free_tagger.pickle",
    tree_taxonomy_file: str = 'full_generated_taxonomy_with_desc.json',
    flat_taxonomy_file: str = 'tnt_clusters_full.pkl',
    cache_prefix: str = "",
    tree_desc_len: int = 50,
    permute_branches: bool = False,
    enabled_taxonomies: Optional[List[str]] = None
):
    print("\n--- Evaluating existing annotations ---")
    output_dir_to_evaluate = bulk_output_dir

    # Load ground truth data if it exists
    ground_truth_df = None
    ground_truth_path = Path(ground_truth_file)
    if ground_truth_path.exists():
        try:
            ground_truth_df = pd.read_pickle(ground_truth_path)
            
            # Robustly set 'job_id' as a string index to ensure consistency.
            if 'job_id' in ground_truth_df.columns:
                # If job_id is a column, set it as index.
                ground_truth_df.set_index('job_id', inplace=True)
            
            if ground_truth_df.index.name == 'job_id':
                # Ensure the index is of type string for matching with annotation job_ids.
                ground_truth_df.index = ground_truth_df.index.astype(str)
                
                # Check if any of the required columns for evaluation exist.
                required_cols = ['canonical_label', 'canonical_label_taxonomy']
                if not any(col in ground_truth_df.columns for col in required_cols):
                    print(f"Warning: Ground truth file {ground_truth_path} is missing all required columns for evaluation (e.g., 'canonical_label').")
                    ground_truth_df = None
                else:
                    print(f"Successfully loaded ground truth data from {ground_truth_path}")
            else:
                # If no 'job_id' can be found as a column or index, we cannot use it.
                print(f"Warning: Ground truth file {ground_truth_path} is missing a 'job_id' column or index. Skipping ground truth evaluation.")
                ground_truth_df = None
        except Exception as e:
            print(f"Warning: Could not load or process ground truth file {ground_truth_path}. Error: {e}. Skipping ground truth evaluation.")
            ground_truth_df = None
    else:
        print(f"Info: Ground truth file not found at {ground_truth_path}. Skipping ground truth evaluation.")

    # Load ID->Title maps for hallucination check, creating the ESCO one if needed.
    create_correct_esco_title_map(DATA_DIR)
    id_to_title_maps = load_id_to_title_maps(
        DATA_DIR, 
        tree_taxonomy_file=tree_taxonomy_file,
        flat_taxonomy_file=flat_taxonomy_file,
        cache_prefix=cache_prefix,
        tree_desc_len=tree_desc_len,
        permute_branches=permute_branches,
        enabled_taxonomies=enabled_taxonomies
    )

    # Load annotations and prepare data for DataFrame and metrics
    all_annotations, dataframe_rows = load_and_parse_annotations(output_dir_to_evaluate)
    
    evaluate_annotations(
        output_dir_to_evaluate, 
        dataframe_rows,
        ground_truth_df=ground_truth_df, 
        id_to_title_maps=id_to_title_maps
    )


def test_tree_reformatting(config: Dict[str, Any]):
    """
    Loads and reformats the tree taxonomy to test string generation,
    verifying that permutation works correctly while preserving hierarchy.
    """
    print("\n--- Testing Tree Taxonomy Reformatting and Permutation ---")
    tree_taxonomy_file = config.get("tree_taxonomy_file")
    if not tree_taxonomy_file or not Path(tree_taxonomy_file).exists():
        print(f"Error: Tree taxonomy file not found at '{tree_taxonomy_file}'")
        return

    title2desc_file = "title2desc.pkl"
    if not Path(title2desc_file).exists():
        print(f"Error: Required file '{title2desc_file}' not found.")
        return
        
    try:
        print(f"Loading tree taxonomy from: {tree_taxonomy_file}")
        with open(tree_taxonomy_file, "r", encoding="utf-8") as f:
            json_string = f.read()
            taxonomy_data_loaded = json.loads(json_string)
            taxonomy_data = {k:v for k,v in taxonomy_data_loaded.items() if int(k)<3}

        print(f"Loading title-to-description map from: {title2desc_file}")
        with open(title2desc_file, "rb") as f:
            title2desc = pickle.load(f)

        # --- 1. Generate Sorted (canonical) version ---
        print("\n1. Generating canonical (sorted) taxonomy string...")
        sorted_str, _ = reformat_tree_taxonomy_for_llm(
            taxonomy_data, 
            title2desc, 
            desc_len=-1,
            permute_branches=False
        )
        print("\n--- START: Canonical String (first 1000 chars) ---")
        print(sorted_str[:1000])
        print("--- END: Canonical String ---\n")

        # --- 2. Generate Permuted version ---
        print("\n2. Generating permuted (shuffled) taxonomy string...")
        permuted_str, _ = reformat_tree_taxonomy_for_llm(
            taxonomy_data, 
            title2desc, 
            desc_len=-1,
            permute_branches=True
        )
        print("\n--- START: Permuted String (first 1000 chars) ---")
        print(permuted_str[:1000])
        print("--- END: Permuted String ---\n")

        # --- 3. Verify Permutation ---
        print("\n3. Verifying permutation...")
        if sorted_str != permuted_str:
            print("SUCCESS: The permuted string is different from the sorted string.")
        else:
            print("FAILURE: The permuted string is identical to the sorted string. Permutation is not working.")
            return

        # --- 4. Verify Hierarchy Preservation ---
        print("\n4. Verifying hierarchy preservation (by checking a sample parent-child structure)...")
        
        # Find a suitable parent node with children to inspect. Let's try the highest non-leaf level.
        levels = sorted([int(k) for k in taxonomy_data.keys() if k != "0"], reverse=True)
        parent_to_inspect = None
        if levels:
            level_key = str(levels[0])
            for item in taxonomy_data.get(level_key, []):
                if item.get("kids"):
                    parent_to_inspect = item
                    break
        
        if not parent_to_inspect:
            print("Could not find a suitable parent node with children to inspect for this test.")
            return
            
        parent_title = parent_to_inspect["title"]
        child_titles = set(parent_to_inspect["kids"])
        print(f"\nInspecting parent: '{parent_title}'")
        print(f"Expected children ({len(child_titles)}): {child_titles}")

        def extract_children_for_parent(text_blob, parent_title):
            """Helper to find a parent and extract its immediate children from the formatted string."""
            found_children = set()
            lines = text_blob.split('\n')
            
            # Find the parent and its indentation level
            for i, line in enumerate(lines):
                if f"Title: {parent_title}" in line:
                    parent_indent = len(line) - len(line.lstrip(' '))
                    
                    # Now look for children on subsequent lines with more indentation
                    for j in range(i + 1, len(lines)):
                        child_line = lines[j]
                        if not child_line.strip(): continue
                        child_indent = len(child_line) - len(child_line.lstrip(' '))
                        
                        if child_indent > parent_indent:
                            # It's a descendant. If it's a direct child, its indent should be exactly 2 more.
                            if child_indent == parent_indent + 2:
                                 try:
                                     child_title = child_line.split("Title: ")[1].split(" - Description:")[0].strip()
                                     found_children.add(child_title)
                                 except IndexError:
                                     print(f"Warning: Could not parse title from line: {child_line}")
                        else:
                            # We've left the parent's subtree
                            break
                    break
            return found_children

        sorted_children = extract_children_for_parent(sorted_str, parent_title)
        permuted_children = extract_children_for_parent(permuted_str, parent_title)

        print(f"\nChildren found under '{parent_title}' in CANONICAL string ({len(sorted_children)}): {sorted_children}")
        print(f"Children found under '{parent_title}' in PERMUTED string ({len(permuted_children)}): {permuted_children}")

        if child_titles == sorted_children and child_titles == permuted_children:
            print("\nSUCCESS: The same set of children was found under the sample parent in both strings. Hierarchy is preserved.")
        else:
            print("\nFAILURE: The set of children found does not match the expected children. Hierarchy may be broken.")
            if child_titles != sorted_children:
                print(f"  Mismatch in CANONICAL string. Missing: {child_titles - sorted_children}, Extra: {sorted_children - child_titles}")
            if child_titles != permuted_children:
                 print(f"  Mismatch in PERMUTED string. Missing: {child_titles - permuted_children}, Extra: {permuted_children - child_titles}")

    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")


def run_workflow(config: Dict[str, Any], run_annotations: bool = False, run_evaluation: bool = True, sample_run: bool = False):
    """
    Executes the main workflow based on the provided configuration.
    """
    base_output_dir = Path(config["output_dir"])
    base_output_dir.mkdir(parents=True, exist_ok=True)

    permute_branches = config.get("permute_branches", False)

    # If permutation is enabled, we must delete any existing permuted cache
    # to ensure a new, different permutation is generated for this run.
    if permute_branches:
        print("\nPermutation is enabled. Deleting previous permuted cache to ensure a new permutation is generated.")
        prefix = f"{config.get('cache_prefix', '')}_" if config.get('cache_prefix') else ""
        tree_taxonomy_stem = Path(config["tree_taxonomy_file"]).stem
        tree_desc_len = config.get("tree_desc_len", 50)
        desc_len_suffix = f"desc{'full' if tree_desc_len < 0 else tree_desc_len}"
        permute_suffix = "_permuted"

        tree_id_map_path = DATA_DIR / f"{prefix}{tree_taxonomy_stem}_{desc_len_suffix}{permute_suffix}_id_to_title_map.json"
        tree_str_path = DATA_DIR / f"{prefix}{tree_taxonomy_stem}_{desc_len_suffix}{permute_suffix}_taxonomy_str.pkl"
        
        try:
            if tree_id_map_path.exists():
                os.remove(tree_id_map_path)
                print(f"  - Removed old permuted map cache: {tree_id_map_path.name}")
            if tree_str_path.exists():
                os.remove(tree_str_path)
                print(f"  - Removed old permuted string cache: {tree_str_path.name}")
        except OSError as e:
            print(f"Warning: Error removing cache files: {e}")

    permute_suffix = "_permuted" if permute_branches else ""
    test_data_stem = Path(config["test_data_file"]).stem
    bulk_annotations_dir = base_output_dir / f"bulk_annotations_{test_data_stem}_with_{config['use_column']}{permute_suffix}"

    enabled_llms = config.get("enabled_llms")
    enabled_taxonomies = config.get("enabled_taxonomies")

    if run_annotations:
        print("\n--- Running Step 1: Annotations ---")
        main(
            use_column=config["use_column"], 
            sample=sample_run,
            test_data_file=config["test_data_file"],
            tree_taxonomy_file=config["tree_taxonomy_file"],
            flat_taxonomy_file=config["flat_taxonomy_file"],
            cache_prefix=config["cache_prefix"],
            tree_desc_len=config.get("tree_desc_len", 50),
            bulk_output_dir=bulk_annotations_dir,
            permute_branches=permute_branches,
            enabled_llms=enabled_llms,
            enabled_taxonomies=enabled_taxonomies
        )
    
    if run_evaluation:
        print("\n--- Running Step 2: Evaluation ---")
        eval(
            bulk_output_dir=bulk_annotations_dir,
            ground_truth_file=config["ground_truth_file"],
            tree_taxonomy_file=config["tree_taxonomy_file"],
            flat_taxonomy_file=config["flat_taxonomy_file"],
            cache_prefix=config["cache_prefix"],
            tree_desc_len=config.get("tree_desc_len", 50),
            permute_branches=permute_branches,
            enabled_taxonomies=enabled_taxonomies
        )


if __name__ == "__main__":
    # --- Configuration for the run ---
    # Define configurations for different datasets in this dictionary.
    # This makes it easy to switch between datasets by changing ACTIVE_CONFIG_NAME.
    CONFIGS = {
        "indeed": {
            "output_dir": "../output",
            "cache_prefix": "",
            "use_column": "job_description", 
            "test_data_file": 'indeed_test_data_1165.pkl',
            "tree_taxonomy_file": 'full_generated_taxonomy_with_desc.json',
            "flat_taxonomy_file": 'tnt_clusters_full.pkl',
            "ground_truth_file": "df_test_annotated_by_free_tagger_and_taxonomy.pkl",
            "tree_desc_len": 50,
            "permute_branches": False,
            "enabled_llms": None,
            "enabled_taxonomies": None
        },
        "palestine": {
            "output_dir": "../output",
            "cache_prefix": "palestine",
            "use_column": "job_description_full",
            "test_data_file": 'palestine_test_data_embedding.pkl',
            "tree_taxonomy_file": '../palestine_taxonomy_output/full_generated_taxonomy_with_desc.json',
            "flat_taxonomy_file": 'palestine_tnt_clusters_full.pkl',
            "ground_truth_file": "palestine_df_test_annotated_by_free_tagger_and_taxonomy.pkl",
            "tree_desc_len": -1,
            "permute_branches": False,
            "enabled_llms": None,
            "enabled_taxonomies": None
        },
        "botswana": {
            "output_dir": "../output",
            "cache_prefix": "botswana",
            "use_column": "job_description",
            "test_data_file": 'botswana_test_data_embedding.pkl',
            "tree_taxonomy_file": '../botswana_taxonomy_output/full_generated_taxonomy_with_desc.json',
            "flat_taxonomy_file": 'botswana_tnt_clusters_full.pkl',
            "ground_truth_file": "botswana_df_test_annotated_by_free_tagger_and_taxonomy.pkl",
            "tree_desc_len": -1,
            "permute_branches": False,
            "enabled_llms": None,
            "enabled_taxonomies": None
        }
    }

    # --- Select Active Configuration and Workflow Steps ---
    
    # 1. CHOOSE YOUR CONFIGURATION by setting the name here.
    ACTIVE_CONFIG_NAME = "indeed" 
    # ACTIVE_CONFIG_NAME = "palestine"

    # 2. CHOOSE WHICH STEPS TO RUN.
    # Set this to True to quickly test the tree string formatting and exit.
    TEST_TREE_REFORMATTING = False

    RUN_ANNOTATIONS = False  # Set to True to generate new annotation files from the LLMs.
    RUN_EVALUATION = True   # Set to True to evaluate existing annotation files.
    
    # 3. (Optional) Set to True for a small sample run (will only affect annotations).
    SAMPLE_RUN = False

    # --- Execute Workflow ---
    if ACTIVE_CONFIG_NAME not in CONFIGS:
        raise ValueError(f"Configuration '{ACTIVE_CONFIG_NAME}' not found in CONFIGS. Available options: {list(CONFIGS.keys())}")
    
    selected_config = CONFIGS[ACTIVE_CONFIG_NAME]
    print(f"\nRunning workflow with configuration: '{ACTIVE_CONFIG_NAME}'")
    
    if TEST_TREE_REFORMATTING:
        test_tree_reformatting(selected_config)
    else:
        run_workflow(
            config=selected_config,
            run_annotations=RUN_ANNOTATIONS,
            run_evaluation=RUN_EVALUATION,
            sample_run=SAMPLE_RUN
        )

