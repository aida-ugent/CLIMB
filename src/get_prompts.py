import pandas as pd
import json
import pickle
import random
from tree_utils import balanced_sampling
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from prompts import basic_instruction, new_instruction


def assign_leaf_id_to_data(df, leaf_to_rep):
    """
    Assign a leaf id to each data point in the dataframe.
    """
    df['leaf_id'] = df.apply(lambda row: leaf_to_rep[row.name], axis=1)
    return df

# Approximate token count function (1 token ~ 4 characters)
def count_tokens(text: str) -> int:
    return len(text) // 4


def truncate_line(line, remaining_tokens):
    """
    Truncate a line to fit within remaining token limit
    
    Args:
        line: str, the line to truncate
        remaining_tokens: int, number of tokens remaining
        
    Returns:
        str: truncated line that fits within token limit
    """
    words = line.split()
    truncated_words = []
    running_tokens = 0
    
    for word in words:
        word_tokens = count_tokens(word + " ")
        if running_tokens + word_tokens > remaining_tokens:
            break
        truncated_words.append(word)
        running_tokens += word_tokens
    
    return " ".join(truncated_words) + "..." if len(truncated_words) < len(words) else line


def generate_prompt_for_node(node_id, postings, token_limit=127000, instruction=basic_instruction):
    """
    Generate a prompt for a given node (leaf_id) using the provided job postings.
    """
    # Build lines for each job posting.
    # Here we use the dataframe index as a surrogate job id.
    postings_lines = []
    for idx, row in postings.iterrows():
        job_id = idx  # Change this if you have an explicit job id column.
        title = row['job_title']
        description = row['job_description']
        postings_lines.append(f"Job id: {job_id} | Title: {title} | Description: {description}")
    
    # Combine lines into a single string
    postings_text = "\n-----\n".join(postings_lines)
    
    # If text is too long, truncate by iterating line by line.
    if count_tokens(postings_text) > token_limit:
        running_tokens = 0
        truncated_lines = []

        for line in postings_lines:
            line_tokens = count_tokens(line + "\n-----\n")
            if running_tokens + line_tokens > token_limit:
                # Truncate this line instead of dropping it
                truncated_line = truncate_line(line, token_limit - running_tokens)
                truncated_lines.append(truncated_line)
                break
            truncated_lines.append(line)
            running_tokens += line_tokens
        postings_text = "\n".join(truncated_lines)
    
    # Build the final prompt using the desired format.
    # prompt = f"""####
    # You are an expert in job classification. Below is a set of related job postings forming a single occupational category. Extract:  

    # 1. Occupation Title: A standardized title representing the common role.  
    # 2. Key Skills: Relevant skills from the job descriptions.  
    # 3. Representative Job Posting IDs: 3-5 job postings IDs that best represent the category. If there are less than 3 input job postings, just list all their IDs.  

    # Input Format:
    # Node id: <node_id>
    # List of job postings as follows:
    # -----
    #     Job id: <job_id> | Title: <title> | Description: <description> \n

    # Output Format (TSV): 
    # Node id \t Occupation title \t Skills \t Representative job posting ids \n 12345 \t Entry-Level Production Worker \t Counting, Stacking, Reading Gauges, Basic Equipment Operation, Shift Work, Manufacturing Facility Setting, Physical Labor, ... \t 12, 233, 5667, 666, 91111 \n
    # #### \n
    # TEXT\nNode id: {node_id}\n{postings_text}
    
    # """
    prompt = f"""####
    {instruction}
    
    Input Format:
    List of job postings as follows:
    -----
        Job id: <job_id> | Title: <title> | Description: <description> 
    
    ONLY OUTPUT THE OCCUPATION TITLE, NO OTHER TEXT.
    #### \n
    TEXT\n{postings_text}
    """

    return prompt


def mmr_sampling(data_points, data_df, k=30, lambda_param=0.5):
    """
    Perform Maximal Marginal Relevance sampling on job postings.
    
    Args:
        data_points: List of indices of job postings to sample from
        data_df: DataFrame containing job data
        k: Number of samples to select
        lambda_param: Balance between relevance and diversity (0-1)
                     Higher values favor relevance, lower values favor diversity
    
    Returns:
        List of selected job posting indices
    """
    if len(data_points) <= k:
        return data_points
    
    # Get job titles and descriptions for the data points
    job_data = data_df.iloc[data_points]
    
    # Create a simple text representation for each job (title + description)
    job_texts = [f"{row['job_title']} {row['job_description']}" for _, row in job_data.iterrows()]
    
    # Use TF-IDF to convert text to vectors
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(job_texts)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Calculate average similarity to get a relevance score for each document
    avg_similarities = similarity_matrix.mean(axis=1)
    
    # MMR algorithm
    selected_indices = []
    remaining_indices = list(range(len(data_points)))
    
    # Select the most relevant document first
    first_idx = avg_similarities.argmax()
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Iteratively select the rest
    for _ in range(min(k-1, len(data_points)-1)):
        mmr_scores = []
        
        for idx in remaining_indices:
            # Relevance term (similarity to query/centroid)
            relevance = avg_similarities[idx]
            
            # Diversity term (negative of maximum similarity to already selected items)
            max_similarity = max([similarity_matrix[idx, selected_idx] for selected_idx in selected_indices])
            diversity = -max_similarity
            
            # MMR score
            mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
            mmr_scores.append(mmr_score)
        
        # Select the item with the highest MMR score
        next_idx = remaining_indices[np.argmax(mmr_scores)]
        selected_indices.append(next_idx)
        remaining_indices.remove(next_idx)
    
    # Map back to original data point indices
    selected_data_points = [data_points[idx] for idx in selected_indices]
    random.shuffle(selected_data_points)
    return selected_data_points


def get_truncated_prompts(data_df, root=None, k=30, use_mmr=True, lambda_param=0.5, instruction=basic_instruction, column_description='job_description', use_mapping=None):
    """
    Generate prompts with improved sampling using MMR if specified.
    
    Args:
        data_df: DataFrame containing job data
        root: Root node of the tree
        k: Number of samples per node
        use_mmr: Whether to use MMR sampling (True) or balanced sampling (False)
        lambda_param: MMR parameter balancing relevance vs. diversity
    """
    if not use_mapping:
        assert root is not None, "Root node is required if use_mapping is False"
        if use_mmr:
            # First get all data points per node using balanced sampling
            sampled_data = balanced_sampling(root, k*2)  # Get more samples initially
            
            # Then apply MMR to each node's samples
            mmr_sampled_data = {}
            for node_id, job_idx in sampled_data.items():
                mmr_sampled_data[node_id] = mmr_sampling(job_idx, data_df, k, lambda_param)
            sampled_data = mmr_sampled_data
        else:
            sampled_data = balanced_sampling(root, k)
    else:
        print(f"Using mapping file: {use_mapping}")
        all_data = pickle.load(open(use_mapping, 'rb'))
        # keep k samples per node
        sampled_data = {}
        for node_id, job_idx in all_data.items():
            # shuffle the job_idx
            random.shuffle(job_idx)
            sampled_data[node_id] = job_idx[:k]
    
    prompts = {}
    for node_id, job_idx in sampled_data.items():
        prompts[node_id] = {}
        prompts[node_id]['sampled_job_idx'] = job_idx
        prompts[node_id]['sampled_job_titles'] = data_df.iloc[job_idx]['job_title'].tolist()
        postings_lines = []
        for id in job_idx:
            job_id = id 
            title = data_df.iloc[id]['job_title']
            description = data_df.iloc[id][column_description]
            line = f"Job id: {job_id} | Title: {title} | Description: {description}"
            # count tokens of the line
            num_tokens = count_tokens(line)
            if num_tokens > 4000:
                # truncate the description to 2500 words
                words = description.split()
                description = ' '.join(words[:2500])
                line = f"Job id: {job_id} | Title: {title} | Description: {description}"
            postings_lines.append(line)
        postings_text = "\n-----\n".join(postings_lines)
        prompt = f"""
        {instruction}
-----
Input Format:
List of job postings as follows:
-----
    Job id: <job_id> | Title: <title> | Description: <description> 
#### \n
{postings_text}
        """
        prompts[node_id]['prompt'] = prompt
        if count_tokens(prompt) > 127000:
            print(node_id)
            print(prompt)
            print('-'*100)
        
    return prompts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/indeed_data_embeddings_deduplicated.pkl')
    parser.add_argument('--output_path', type=str, default='../data/prompts_test.json')
    parser.add_argument('--tree_path', type=str, default='../tree.pkl')
    parser.add_argument('--k', type=int, default=30)
    parser.add_argument('--use_mmr', type=bool, default=False)
    parser.add_argument('--lambda_param', type=float, default=0.5)
    parser.add_argument('--instruction', type=str, default='basic')
    parser.add_argument('--column_description', type=str, default='job_description')
    args = parser.parse_args()

    df = pickle.load(open(args.data_path, 'rb'))
    tree = pickle.load(open(args.tree_path, 'rb'))
    if args.instruction == 'new':
        instruction = new_instruction
    else:
        instruction = basic_instruction
    prompts = get_truncated_prompts(df, tree, k=args.k, use_mmr=args.use_mmr, lambda_param=args.lambda_param, instruction=instruction, column_description=args.column_description)
    print(len(prompts))
    print(prompts[23284]['sampled_job_titles'])
    print('-'*100)
    print(prompts[23285]['sampled_job_titles'])
    print('-'*100)
    print(prompts[22307]['sampled_job_titles'])
    # save the prompts
    # with open(args.output_path, 'w') as f:
    #     json.dump(prompts, f)



