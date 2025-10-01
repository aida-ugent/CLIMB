#!/usr/bin/env python3

import os
import openai
import pickle

import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def add_prompts(df):
    """
    Add classification prompts to a dataframe as a new column.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing a 'text' column with job posting chunks
    
    Returns:
    pandas.DataFrame: The original dataframe with a new 'prompt' column
    """
    if 'text' not in df.columns:
        raise ValueError("DataFrame must contain a 'text' column")
    
    # Define the prompt template
    template = """You are an expert job posting analyst. Your task is to determine if a given text chunk from a job posting is relevant for identifying the job's standard occupation title.

    **Instructions:**

    Read the provided text chunk carefully.  Determine if the chunk contains information directly related to the *nature of the work performed* in the job.

    **Relevance Criteria:**

    A text chunk is **relevant (Y)** if it includes information about *any* of the following:

    *   **Job Duties & Responsibilities:**  Specific tasks the employee will perform (e.g., "Write code," "Manage budgets," "Conduct research").
    *   **Required Skills (Technical & Soft):**  Abilities needed to perform the job (e.g., "Python proficiency," "Communication skills," "Project management").
    *   **Qualifications:** Education, certifications, or experience levels required (e.g., "Bachelor's degree," "PMP certification," "5+ years experience").
    *   **Job Title & Description (Specific):**  The formal name of the position and a summary of its purpose (e.g., "Software Engineer," "Marketing Manager," "Responsible for...").
    *   **Tools & Technologies:**  Software, hardware, or methodologies used in the role (e.g., "AWS," "Agile," "CRM").
    *   **Team & Reporting Structure:**  The team the role belongs to and who the role reports to (e.g., "Reports to the Engineering Manager," "Part of the Sales team").

    A text chunk is **irrelevant (N)** if it primarily focuses on:

    *   **Compensation & Benefits:**  Salary, health insurance, vacation, etc.
    *   **Company Culture & Values:**  Work environment, mission, or beliefs.
    *   **Application Process:**  Instructions on how to apply.
    *   **Diversity & Inclusion:**  Statements about equal opportunity.
    *   **Company Information (General):**  History, size, or industry.

    **Output:** Respond with ONLY "Y" or "N" (without explanation).

    **Examples:**
    Input: "Design and implement scalable microservices using Java and AWS."  Output: Y
    Input: "Our comprehensive benefits package includes 401k matching and generous PTO." Output: N
    Input: "Bachelor's degree in Computer Science or related field required." Output: Y
    Input: "Join our inclusive workplace where diversity is celebrated." Output: N

    -------------------
    Classify the following text chunk:
    Input: "{text}"
    Output:"""
    
    # Add prompts as a new column
    df['prompt'] = df['text'].apply(lambda x: template.format(text=x))
    
    return df


def process_job_postings(input_file, output_file):
    print("Loading data...")
    data = pickle.load(open(input_file, 'rb'))

    print("Adding prompts to data...")
    data = add_prompts(data)
    
    # Check if output file exists and load it to resume progress
    if os.path.exists(output_file):
        print(f"Found existing output file. Loading {output_file} to resume...")
        labeled_data = pickle.load(open(output_file, 'rb'))
        
        # Merge the labels from the output file back into our data
        # This ensures we don't process chunks that have already been labeled
        for idx in labeled_data.index:
            if idx in data.index and labeled_data.loc[idx, 'label'] != '':
                data.loc[idx, 'label'] = labeled_data.loc[idx, 'label']
    
    # Create a new column for labels if it doesn't exist
    if 'label' not in data.columns:
        data['label'] = ''
    
    # Process all rows that don't have a label yet
    unlabeled_indices = data[(data['label'] == '') | (data['label'].isna())].index
    total = len(unlabeled_indices)
    
    print(f"Processing {total} unlabeled chunks...")
    
    # Set up OpenAI client
    print("Setting up OpenAI client...")
    client = openai.OpenAI()    

    for i in tqdm(unlabeled_indices):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": data.loc[i, 'prompt']}],
                temperature=0.0,
            )
            data.loc[i, 'label'] = response.choices[0].message.content.strip()
            
            # Save progress every 100 items
            if (i % 100 == 0) and (i > 0):
                print(f"Saving progress... ({i}/{total})")
                pickle.dump(data, open(output_file, 'wb'))
                
        except Exception as e:
            print(f"Error processing index {i}: {e}")
            # Save progress on error
            pickle.dump(data, open(output_file, 'wb'))
    
    print(data.head())
    # Final save
    print("Saving final results...")
    pickle.dump(data, open(output_file, 'wb'))
    print("Processing complete!")


def generate_batch_prompt(df, indices, max_words=1000):
    """
    Generate a prompt for batch classification of multiple text chunks.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing a 'text' column with job posting chunks
    indices (list): List of indices to include in the batch
    max_words (int): Maximum number of words to include from each text chunk
    
    Returns:
    str: A prompt for batch classification
    """
    if 'text' not in df.columns:
        raise ValueError("DataFrame must contain a 'text' column")
    
    # Create the batch prompt
    batch_template = """You are an expert job posting analyst. Your task is to determine if given text chunks from job postings are relevant for identifying the job's standard occupation title.

**Instructions:**

For each text chunk, determine if it contains information directly related to the *nature of the work performed* in the job.

**Relevance Criteria:**

A text chunk is **relevant (Y)** if it includes information about *any* of the following:

*   **Job Duties & Responsibilities:**  Specific tasks the employee will perform
*   **Required Skills (Technical & Soft):**  Abilities needed to perform the job
*   **Qualifications:** Education, certifications, or experience levels required
*   **Job Title & Description (Specific):**  The formal name of the position and its purpose
*   **Tools & Technologies:**  Software, hardware, or methodologies used in the role
*   **Team & Reporting Structure:**  The team the role belongs to and reporting lines

A text chunk is **irrelevant (N)** if it primarily focuses on:

*   **Compensation & Benefits:**  Salary, health insurance, vacation, etc.
*   **Company Culture & Values:**  Work environment, mission, or beliefs
*   **Application Process:**  Instructions on how to apply
*   **Diversity & Inclusion:**  Statements about equal opportunity
*   **Company Information (General):**  History, size, or industry

**Output Format:** 
For each chunk, respond with the chunk ID followed by either "Y" or "N" (without explanation), one per line.
Example: "123: Y" or "456: N"

-------------------
Classify the following text chunks with their ids:

{chunks}

Output:"""
    
    # Create the chunks section
    chunks_text = ""
    for idx in indices:
        # Limit by word count instead of character count
        words = df.loc[idx, 'text'].split()
        truncated_text = " ".join(words[:max_words])
        chunks_text += f"Chunk {idx}: \"{truncated_text}\"\n\n"
    
    # Generate the complete prompt
    prompt = batch_template.format(chunks=chunks_text)
    
    return prompt


def process_batches(input_file, output_file, batch_size=20, max_words=1000, client=None):
    """
    Process job posting chunks in batches and annotate with LLM responses.
    
    Parameters:
    input_file (str): Path to input pickle file containing DataFrame with job posting chunks
    output_file (str): Path to output pickle file to save the labeled DataFrame
    batch_size (int): Number of chunks to process in each API call
    max_words (int): Maximum number of words to include from each text chunk
    client: OpenAI client instance
    
    Returns:
    pandas.DataFrame: The DataFrame with a new 'label' column containing annotations
    """
    print(f"Loading data from {input_file}...")
    df = pickle.load(open(input_file, 'rb'))
    
    if client is None:
        import openai
        client = openai.OpenAI()
        print("OpenAI client initialized.")
    
    # Check if output file exists and load it to resume progress
    if os.path.exists(output_file):
        print(f"Found existing output file. Loading {output_file} to resume...")
        labeled_df = pickle.load(open(output_file, 'rb'))
        
        # Merge the labels from the output file back into our data
        for idx in labeled_df.index:
            if idx in df.index and not pd.isna(labeled_df.loc[idx, 'label']):
                df.loc[idx, 'label'] = labeled_df.loc[idx, 'label']
    
    # Initialize label column if it doesn't exist
    if 'label' not in df.columns:
        df['label'] = None
    
    # Get indices of rows that need labeling
    indices_to_process = df[(df['label'].isna()) | (df['label'] == '')].index.tolist()
    total = len(indices_to_process)
    
    print(f"Processing {total} unlabeled chunks in batches of {batch_size}...")
    
    # Process in batches
    for i in tqdm(range(0, len(indices_to_process), batch_size)):
        batch_indices = indices_to_process[i:i+batch_size]
        prompt = generate_batch_prompt(df, batch_indices, max_words)
            
        try:
            # Call the API
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            
            # Parse the response
            result = response.choices[0].message.content
            for line in result.strip().split('\n'):
                if ':' in line:
                    try:
                        chunk_id, label = line.split(':', 1)
                        chunk_id = int(chunk_id.strip().replace('Chunk ', ''))
                        label = label.strip()
                        if label in ['Y', 'N']:
                            df.loc[chunk_id, 'label'] = label
                    except (ValueError, KeyError) as e:
                        print(f"Error parsing line: {line}, {e}")
            
            # Save progress every 500 batches
            if (i // batch_size) % 500 == 0 and i > 0:
                print(f"Saving progress... (processed {i+len(batch_indices)}/{total} chunks)")
                pickle.dump(df, open(output_file, 'wb'))
                
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {e}")
            # Save progress on error
            pickle.dump(df, open(output_file, 'wb'))
    
    # Final save
    print("Saving final results...")
    pickle.dump(df, open(output_file, 'wb'))
    print("Processing complete!")
    
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process job postings')
    parser.add_argument('--input', default='data/job_postings_chunks.pkl', help='Path to input pickle file')
    parser.add_argument('--output', default='data/job_postings_chunks_labeled_new.pkl', help='Path to output pickle file')
    parser.add_argument('--batch-size', type=int, default=30, help='Number of chunks to process in each API call') # 50?
    parser.add_argument('--max-words', type=int, default=750, help='Maximum number of words of text chunks') # roughly 1000 tokens
    args = parser.parse_args()
    
    process_batches(args.input, args.output, args.batch_size, args.max_words)