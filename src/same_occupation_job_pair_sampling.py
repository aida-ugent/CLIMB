from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

df_embeddings = pd.read_pickle('../tmp/indeed_data_embeddings_annotated.pkl')


output_file = Path('../tmp/indeed_same_job_pos_neg_samples.pkl')

if not output_file.exists():    
    embeddings = np.vstack(df_embeddings['bge_m3_emb'].values)
    embeddings_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / embeddings_norm
    assert np.allclose(np.sum(embeddings ** 2, axis=1), 1)

    np.random.seed(42)
    pairs = {}
    for i in tqdm(range(embeddings.shape[0])):
        embedding = embeddings[i]
        
        scores = np.dot(embedding, embeddings.T) # (embeddings.shape[0], )    
        indices = np.argsort(scores)[::-1]
        
        sampled_indices, sample_types = [], []
        # positive pair sample 2 from top 21
        positive_indices = np.random.choice(indices[1:21], 2, replace=False)
        sampled_indices.extend(positive_indices.tolist())
        sample_types.extend(['positive'] * len(positive_indices))
        # hard negative, sample 2 from top 21 to 100
        hard_negative_indices = np.random.choice(indices[21:101], 1, replace=False)
        sampled_indices.extend(hard_negative_indices.tolist())
        sample_types.extend(['hard_negative'] * len(hard_negative_indices))
        # easy negative, sample 2 from rest
        easy_negative_indices = np.random.choice(indices[101:], 1, replace=False)
        sampled_indices.extend(easy_negative_indices.tolist())
        sample_types.extend(['easy_negative'] * len(easy_negative_indices))

        for idx, sample_type in zip(sampled_indices, sample_types):
            if (i, idx) in pairs or (idx, i) in pairs:
                continue
            pairs[(i, idx)] = {
                'sample_type': sample_type,
                'score': float(scores[idx])
           }

    df_pairs = []
    for pair, data in pairs.items():
        entry = {
            'id_a': pair[0],
            'id_b': pair[1],
            'title_a': df_embeddings.iloc[pair[0]]['job_title'],
            'description_a': df_embeddings.iloc[pair[0]]['job_description'].replace('\n', ' '),
            'title_b': df_embeddings.iloc[pair[1]]['job_title'],
            'description_b': df_embeddings.iloc[pair[1]]['job_description'].replace('\n', ' '),        
        }
        entry.update(data)
        df_pairs.append(entry)

    df_pairs = pd.DataFrame(df_pairs)

    df_pairs.to_pickle(output_file)
else:
    df_pairs = pd.read_pickle(output_file)


system_prompt = """You are a specialized AI assistant for analyzing job advertisements. Your task is to determine if two job postings represent the same occupation category, regardless of seniority level, company, or minor variations in responsibilities. Focus on identifying whether both jobs would reasonably be classified under the same specific occupation label. Provide clear, consistent evaluations based solely on the occupation similarity between the job advertisements."""

user_prompt_template = """**Task:** Determine if two job advertisements describe substantially similar occupations for clustering purposes.

**Context:** Your assessment will help train a classifier that groups job postings into occupation-specific clusters. The goal is to create coherent clusters that can be easily labeled with specific occupation titles.

**Compare these aspects:**
1. **Occupation Category:** Do both jobs belong to the same specific occupation category?
2. **Core Skills & Knowledge:** Do they require the same fundamental skill set and domain knowledge?
3. **Work Activities:** Do they involve similar day-to-day activities and responsibilities?

**Decision Rule:**
Classify as "SAME_OCCUPATION" if the jobs share enough similarities that they would reasonably be given the same specific occupation label (e.g., both would be classified as "Frontend Developer" or "Financial Analyst").

**Output Format:**
{{"label": "SAME_OCCUPATION"}} or {{"label": "DIFFERENT_OCCUPATION"}}

**Job Advertisements:**
**Job A:**
* **Title:** {job_a_title}
* **Description:** {job_a_description}
**Job B:**
* **Title:** {job_b_title}
* **Description:** {job_b_description}"""
    

import pickle
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

def generate_response(model_name, messages, return_chat_response=False):

    chat_response = client.chat.completions.create(
        model=model_name,
        temperature=0,        
        max_tokens=4096,
        messages=messages,
        stream=False,
    )
    if return_chat_response:
        return chat_response
    else:
        return chat_response.choices[0].message.content


output_file = Path('../tmp/indeed_same_job_pos_neg_samples_annotated.pkl')
if not output_file.exists():
    annotated_pairs = {}
else:
    annotated_pairs = pickle.load(open(output_file, 'rb'))

for rid, row in tqdm(df_pairs.iterrows(), total=len(df_pairs)):
    if (row['id_a'], row['id_b']) in annotated_pairs or (row['id_b'], row['id_a']) in annotated_pairs:
        continue

    job_a_title = row['title_a']
    job_a_description = row['description_a']
    job_b_title = row['title_b']
    job_b_description = row['description_b']
    
    prompt = user_prompt_template.format(
        job_a_title=job_a_title,
        job_a_description=job_a_description,
        job_b_title=job_b_title,
        job_b_description=job_b_description
    )

    response = generate_response(
        model_name="gpt-4o-mini",
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ],
        return_chat_response=True
    )
    annotated_pairs[(row['id_a'], row['id_b'])] = {
        'prompt': prompt,
        'response': response
    }
    pickle.dump(annotated_pairs, open(output_file, 'wb'))