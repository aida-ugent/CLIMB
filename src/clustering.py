import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import pickle
import numpy as np
from glob import glob
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm



# DEVICE = 'mps'
import xgboost as xgb
from joblib import load
import torch
from torch.nn.utils.rnn import pad_sequence
import warnings
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score

# Check if CUDA is available and set specific device
GPU_ID = 0
DEVICE = f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")


CLUSTER_DIR = Path("../tmp/xgbt_clustering")
DATA_DIR = Path("../data/")
# df_train = pickle.load(open(DATA_DIR / 'indeed_train_data_10129.pkl', 'rb'))
# df_test = pickle.load(open(DATA_DIR / 'indeed_test_data_1165.pkl', 'rb'))


# output_path = DATA_DIR / "indeed_train_test_data_embedding.pkl"
# df_data = pickle.load(open(output_path, "rb"))

# df_train = df_train.merge(df_data[['job_id', 'title_qwen3_8b_emb', 'description_qwen3_8b_emb', 'full_text_qwen3_8b_emb', 'title_qwen3_06b_emb', 'description_qwen3_06b_emb', 'full_text_qwen3_06b_emb', 'title_qwen3_4b_emb', 'description_qwen3_4b_emb', 'full_text_qwen3_4b_emb']], on='job_id', how='left')
# df_test = df_test.merge(df_data[['job_id', 'title_qwen3_8b_emb', 'description_qwen3_8b_emb', 'full_text_qwen3_8b_emb', 'title_qwen3_06b_emb', 'description_qwen3_06b_emb', 'full_text_qwen3_06b_emb', 'title_qwen3_4b_emb', 'description_qwen3_4b_emb', 'full_text_qwen3_4b_emb']], on='job_id', how='left')
df_train = pickle.load(open(DATA_DIR / 'botswana_train_data_embedding.pkl', 'rb'))
df_test = pickle.load(open(DATA_DIR / 'botswana_test_data_embedding.pkl', 'rb'))

ALPHA = 0.8


TMP_DIR = Path("../tmp")
CLUSTER_DIR = TMP_DIR / "xgbt_clustering"

model_qwen8b = load(str(CLUSTER_DIR / 'best_gradient_boosting_model_qwen8b.pkl'))


def compute_feature(target_emb, candidate_embs, revert=False):
    if revert:
        return np.vstack([np.hstack([candidate_embs, np.repeat([target_emb], len(candidate_embs), axis=0), candidate_embs - target_emb, candidate_embs * target_emb])])
    else:
        return np.vstack([np.hstack([np.repeat([target_emb], len(candidate_embs), axis=0), candidate_embs, target_emb - candidate_embs, target_emb * candidate_embs])])
    
    
def compute_feature_vectorized(job_embeddings, model, batch_size=100):
    """Memory-efficient computation of all pairwise features"""
    n_jobs = len(job_embeddings)
    embedding_dim = job_embeddings.shape[1]
    
    # Pre-allocate result array
    similarities = np.zeros((n_jobs, n_jobs), dtype=np.float32)
    
    print(f"Computing similarities for {n_jobs} jobs with {embedding_dim}D embeddings...")
    
    # Process in much smaller batches to manage memory
    for i in tqdm(range(0, n_jobs, batch_size)):
        end_i = min(i + batch_size, n_jobs)
        batch_target = job_embeddings[i:end_i]  # (batch_size, embedding_dim)
        
        # Process each target against all candidates in chunks
        for j in range(0, n_jobs, batch_size):
            end_j = min(j + batch_size, n_jobs)
            batch_candidate = job_embeddings[j:end_j]  # (batch_size, embedding_dim)
            
            # Compute features for this target-candidate batch pair
            n_targets = end_i - i
            n_candidates = end_j - j
            
            # Create feature matrix for this small block
            features = []
            for target_emb in batch_target:
                # Compute features for this target against candidate batch
                target_repeated = np.repeat([target_emb], n_candidates, axis=0)
                diff = target_repeated - batch_candidate
                prod = target_repeated * batch_candidate
                
                # Concatenate features: [target, candidate, diff, prod]
                batch_features = np.hstack([target_repeated, batch_candidate, diff, prod])
                features.append(batch_features)
            
            # Stack all features for this block
            features = np.vstack(features)  # (n_targets * n_candidates, 4 * embedding_dim)
            
            # Predict scores
            scores = model.predict(xgb.DMatrix(features))
            scores = scores.reshape(n_targets, n_candidates)
            
            # Store results
            similarities[i:end_i, j:end_j] = scores
    
    return similarities


def compute_feature_vectorized_gpu(job_embeddings, model, batch_size=50, device=f'cuda:{GPU_ID}'):
    """GPU-accelerated computation of all pairwise features with memory management"""
    n_jobs = len(job_embeddings)
    embedding_dim = job_embeddings.shape[1]
    
    # Move embeddings to GPU
    job_embeddings_gpu = torch.tensor(job_embeddings, dtype=torch.float32, device=device)
    
    # Pre-allocate result array on CPU (XGBoost needs CPU arrays)
    similarities = np.zeros((n_jobs, n_jobs), dtype=np.float32)
    
    print(f"Computing similarities for {n_jobs} jobs with {embedding_dim}D embeddings on {device}...")
    print(f"Using batch size: {batch_size}")
    
    # Clear GPU cache before starting
    torch.cuda.empty_cache()
    
    # Process in small batches to manage GPU memory
    with torch.no_grad():
        for i in tqdm(range(0, n_jobs, batch_size)):
            end_i = min(i + batch_size, n_jobs)
            
            try:
                batch_target = job_embeddings_gpu[i:end_i]  # (batch_size, embedding_dim)
                
                # Process in sub-batches for candidates to avoid memory issues
                candidate_batch_size = min(batch_size * 10, n_jobs)  # Process more candidates at once
                
                for j in range(0, n_jobs, candidate_batch_size):
                    end_j = min(j + candidate_batch_size, n_jobs)
                    batch_candidates = job_embeddings_gpu[j:end_j]
                    
                    # Vectorized feature computation on GPU for this sub-batch
                    target_expanded = batch_target.unsqueeze(1)  # (batch_size, 1, embedding_dim)
                    candidate_expanded = batch_candidates.unsqueeze(0)  # (1, candidate_batch_size, embedding_dim)
                    
                    target_repeated = target_expanded.expand(end_i-i, end_j-j, embedding_dim)
                    candidate_repeated = candidate_expanded.expand(end_i-i, end_j-j, embedding_dim)
                    
                    # Compute all feature components
                    diff = target_repeated - candidate_repeated
                    prod = target_repeated * candidate_repeated
                    
                    # Concatenate features: [target, candidate, diff, prod]
                    features = torch.cat([target_repeated, candidate_repeated, diff, prod], dim=2)
                    
                    # Reshape for prediction and move to CPU
                    features_reshaped = features.view(-1, 4 * embedding_dim).cpu().numpy()
                    
                    # Predict scores - let XGBoost handle GPU/CPU internally
                    scores = model.predict(xgb.DMatrix(features_reshaped))
                    
                    scores = scores.reshape(end_i-i, end_j-j)
                    similarities[i:end_i, j:end_j] = scores
                    
                    # Clear intermediate tensors
                    del target_expanded, candidate_expanded, target_repeated, candidate_repeated
                    del diff, prod, features, features_reshaped
                    
                # Clear GPU cache after each target batch
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️ GPU memory error at batch {i}: {e}")
                    print(f"💡 Try reducing batch_size from {batch_size} to {batch_size//2}")
                    torch.cuda.empty_cache()
                    raise e
                else:
                    raise e
    
    return similarities


# 8B model
similarities_path = CLUSTER_DIR / "botswana_train_qwen3_8b_similarities.pkl"
if not similarities_path.exists():
    print('Start computing similarities using Qwen3 8b\n========================================')
    job_embeddings = np.concatenate([np.vstack(df_train['title_qwen3_8b_emb'].values)*ALPHA, np.vstack(df_train['description_qwen3_8b_emb'].values)*(1-ALPHA)], axis=1)

    print(job_embeddings.shape)

    similarities = compute_feature_vectorized_gpu(job_embeddings, model_qwen8b, device=DEVICE)
    pickle.dump(similarities, open(CLUSTER_DIR / "botswana_train_qwen3_8b_similarities.pkl", 'wb'))
    print("✅ Qwen3 8B similarities saved!")
else:
    similarities = pickle.load(open(CLUSTER_DIR / "botswana_train_qwen3_8b_similarities.pkl", 'rb'))
    print("📁 Loaded existing Qwen3 8B similarities")

symmetrized_similarities = (similarities + similarities.T) / 2
print(symmetrized_similarities.shape)
print('Start clustering using Qwen3 8b\n========================================')
af = AffinityPropagation(affinity='precomputed', random_state=0)
af.fit(symmetrized_similarities)
# save the model
pickle.dump(af, open(CLUSTER_DIR / "botswana_ap_model_qwen3_8b.pkl", 'wb'))

labels = af.labels_
n_clusters_ = len(af.cluster_centers_indices_)
print(f"Estimated number of clusters: {n_clusters_}")

D = 1.01 - symmetrized_similarities
np.fill_diagonal(D, 0)
score = silhouette_score(D, labels, metric='precomputed')
print(f"silhouette score: {score:.6f}")
print("🎉 All models completed! 🎉")