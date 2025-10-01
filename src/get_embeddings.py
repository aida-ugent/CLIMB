import pandas as pd
import pickle
from glob import glob
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.nn.utils.rnn import pad_sequence
import warnings
from pathlib import Path


def de_duplicate(df):
    # remove duplicates from df
    print(f'Removing duplicates from {len(df)} rows')
    df = df.drop_duplicates(subset=['job_description'], keep='first')
    print(f'After removing duplicates, {len(df)} rows left')
    return df


def preprocess_text(text):
    # heuristically remove the first paragraph and the last paragraph
    text = text.split('\n\n')[1:-1]
    text = '\n\n'.join(text)
    return text


def get_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model(**inputs)
    # Move output back to CPU for numpy conversion
    return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()


# Then remove near-duplicates using embeddings similarity
def find_similar_rows(embeddings_array, similarity_threshold=0.98):
    """Find indices of rows with very similar embeddings"""
    # Calculate cosine similarity matrix
    similarities = embeddings_array @ embeddings_array.T
    # Normalize by vector magnitudes
    norms = np.linalg.norm(embeddings_array, axis=1)
    similarities = similarities / (norms[:, None] * norms[None, :])
    
    # Find pairs of similar documents (excluding self-similarity)
    similar_pairs = np.where((similarities > similarity_threshold) & (similarities < 1.0))
    
    # Keep track of indices to remove
    to_remove = set()
    for idx1, idx2 in zip(*similar_pairs):
        if idx1 < idx2:  # Only process each pair once
            to_remove.add(idx2)  # Keep the first occurrence
            
    return list(to_remove)


def get_embedding_df(df, model_name='BAAI/bge-m3', column_description='job_description', batch_size=128, device='cuda:2', remove_emb_duplicates=True):
    # df = de_duplicate(df)
    # reset index
    df = df.reset_index(drop=True)
    model = AutoModel.from_pretrained(model_name)
    # if model_name == 'BAAI/bge-m3':
    #     model.max_seq_length = 4096
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    embeddings = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        batch_texts = [title + ' ' + description for title, description in zip(batch['job_title'], batch[column_description])]
        batch_embeddings = [get_embeddings(text, model, tokenizer) for text in batch_texts]
        embeddings.extend(batch_embeddings)
    df['title_description_emb'] = embeddings
    # Convert embeddings to numpy array
    embeddings_array = np.vstack(df['title_description_emb'].values)

    if remove_emb_duplicates:
        # Find indices to remove
        indices_to_remove = find_similar_rows(embeddings_array)
        # Remove near-duplicate rows
        df = df.drop(indices_to_remove).reset_index(drop=True)

        print(f"Removed {len(indices_to_remove)} near-duplicate entries")
    print(f"Final dataset size: {len(df)}")# ... existing code ...
    # normalize embeddings
    df['title_description_emb'] = df['title_description_emb'].apply(lambda x: x / np.linalg.norm(x))
    return df



def get_embeddings_flash_attention(texts, model, tokenizer, max_length=4096, batch_size=1):
    """
    Get embeddings using FlashAttention2 with proper padding handling.
    
    Args:
        texts: List of text strings or single text string
        model: Pre-loaded model
        tokenizer: Pre-loaded tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size for processing
    
    Returns:
        numpy array of embeddings
    """
    if isinstance(texts, str):
        texts = [texts]
    
    device = next(model.parameters()).device
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # For FlashAttention2, we need to handle padding carefully
        # Option 1: Process each text individually (no padding issues)
        if hasattr(model.config, '_attn_implementation') and model.config._attn_implementation == 'flash_attention_2':
            batch_embeddings = []
            for text in batch_texts:
                # Process individually to avoid padding issues
                inputs = tokenizer(text, truncation=True, return_tensors='pt', max_length=max_length)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Use mean pooling over the sequence dimension
                    embedding = outputs.last_hidden_state.mean(dim=1)
                    batch_embeddings.append(embedding)
            
            batch_embeddings = torch.cat(batch_embeddings, dim=0)
        else:
            # Standard batched processing with padding
            inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                             return_tensors='pt', max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        
        embeddings.append(batch_embeddings.cpu().numpy())
    
    return np.vstack(embeddings)


def load_model_with_fallback(model_name, device='cuda', try_flash_attention=True):
    """
    Load model with FlashAttention2 if available, otherwise fallback to standard attention.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if try_flash_attention:
        try:
            model = AutoModel.from_pretrained(
                model_name, 
                attn_implementation="flash_attention_2", 
                torch_dtype=torch.float16
            )
            print("✅ Successfully loaded model with FlashAttention2")
        except ImportError as e:
            print(f"⚠️  FlashAttention2 not available: {e}")
            print("📦 Install with: pip install flash-attn --no-build-isolation")
            print("🔄 Falling back to standard attention...")
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    else:
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    
    model = model.to(device)
    return model, tokenizer


def get_embeddings_df_new(df, model_name='Qwen/Qwen3-Embedding-8B', device='cuda:1', batch_size=32, max_length=4096, output_path='data/botswana_train_test_data_embedding.pkl'):
    output_path = Path(output_path)
    if not output_path.exists():
        model, tokenizer = load_model_with_fallback(model_name, device, try_flash_attention=True)
        # tokenizer.padding_side  = 'left'

        model = model.to(device)

        batch_size = 32
        title_embeddings = []
        description_embeddings = []
        full_text_embeddings = []
        for i in tqdm(range(0, len(df), batch_size)):
            batch = df.iloc[i:i+batch_size]
                                    
            batch_titles = batch['title'].tolist()
            batch_embeddings_title = [get_embeddings_flash_attention(text, model, tokenizer, max_length=max_length) for text in batch_titles]
            title_embeddings.extend(batch_embeddings_title)

            batch_descriptions = batch['text'].tolist()
            batch_embeddings_description = [get_embeddings_flash_attention(text, model, tokenizer, max_length=max_length) for text in batch_descriptions]
            description_embeddings.extend(batch_embeddings_description)

            batch_full_texts = [title + '\n' + description for title, description in zip(batch['title'], batch['text'])]
            batch_embeddings_full_texts = [get_embeddings_flash_attention(text, model, tokenizer, max_length=max_length) for text in batch_full_texts]
            full_text_embeddings.extend(batch_embeddings_full_texts)
            
        df['title_qwen3_8b_emb'] = title_embeddings
        df['description_qwen3_8b_emb'] = description_embeddings
        df['full_text_qwen3_8b_emb'] = full_text_embeddings
        print(df.head(1))
        df.to_pickle(output_path)
    else:
        df = pickle.load(open(output_path, "rb"))
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='BAAI/bge-large-en-v1.5')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_path', type=str, default='../data/indeed_data.pkl')
    parser.add_argument('--output_path', type=str, default='../data/indeed_data_embeddings.pkl')
    parser.add_argument('--device', type=str, default='cuda:2')
    args = parser.parse_args()
    df = pd.read_pickle(args.data_path)
    # df = get_embedding_df(df, args.model_name, args.batch_size, args.device)
    # df.to_pickle(args.output_path)
    MAX_LENGTH = 4096
    df = get_embeddings_df_new(df, model_name=args.model_name, device=args.device, batch_size=args.batch_size, max_length=MAX_LENGTH, output_path=args.output_path)
    df.to_pickle(args.output_path)


