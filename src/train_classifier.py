#!/usr/bin/env python
# coding: utf-8
import os
# Set PyTorch CUDA memory allocation configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import argparse
import os
from tqdm import tqdm
import pickle


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_model(model, train_loader, val_loader, epochs, learning_rate, warmup_steps, device='cuda:0', log_steps=50, output_dir="./models", multi_gpu=False, gradient_accumulation_steps=1, fp16=False):
    # Wrap model with DataParallel if using multiple GPUs
    if multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = torch.nn.DataParallel(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize scaler for mixed precision training if fp16 is enabled
    scaler = torch.cuda.amp.GradScaler() if fp16 and torch.cuda.is_available() else None
    if fp16 and torch.cuda.is_available():
        print("Using mixed precision training (FP16)")
    
    best_val_loss = float('inf')
    global_step = 0
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_classifier")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        step_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
        
        for step, batch in enumerate(train_pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Only zero gradients at the beginning of accumulation steps
            if (step % gradient_accumulation_steps == 0) or (gradient_accumulation_steps == 1):
                optimizer.zero_grad()
            
            # Use autocast for mixed precision training if fp16 is enabled
            if fp16 and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    # If using gradient accumulation, scale the loss
                    loss = outputs.loss

                    # When using DataParallel, make sure loss is a scalar
                    if multi_gpu and torch.cuda.device_count() > 1:
                        loss = loss.mean()

                    if gradient_accumulation_steps > 1:
                        loss = loss / gradient_accumulation_steps
                
                # Scale the loss and do backward pass with scaler
                scaler.scale(loss).backward()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                # If using gradient accumulation, scale the loss
                loss = outputs.loss

                # When using DataParallel, make sure loss is a scalar
                if multi_gpu and torch.cuda.device_count() > 1:
                    loss = loss.mean()

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                
                loss.backward()
            
            train_loss += loss.item() * (1 if gradient_accumulation_steps == 1 else gradient_accumulation_steps)
            step_loss += loss.item() * (1 if gradient_accumulation_steps == 1 else gradient_accumulation_steps)
            
            # Only update weights after accumulating enough gradients or if not using accumulation
            if ((step + 1) % gradient_accumulation_steps == 0) or (step + 1 == len(train_loader)) or (gradient_accumulation_steps == 1):
                if fp16 and torch.cuda.is_available():
                    # Update with scaler for fp16

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                
                # Only increment global step after a full update
                global_step += 1
                
                train_pbar.set_postfix({'loss': loss.item() * (1 if gradient_accumulation_steps == 1 else gradient_accumulation_steps)})
                
                # Print loss every log_steps
                if global_step % log_steps == 0:
                    avg_step_loss = step_loss / min(log_steps, (step + 1) // gradient_accumulation_steps + 1)
                    print(f"Epoch {epoch+1}/{epochs}, Step {global_step}, Average loss over last {log_steps} steps: {avg_step_loss:.4f}")
                    step_loss = 0
                    
                    # Run validation every log_steps to check for best model
                    if val_loader is not None:
                        model.eval()
                        val_step_loss = 0
                        val_predictions = []
                        val_true_labels = []
                        
                        with torch.no_grad():
                            for val_batch in val_loader:
                                input_ids = val_batch['input_ids'].to(device)
                                attention_mask = val_batch['attention_mask'].to(device)
                                labels = val_batch['label'].to(device)
                                
                                # Use autocast for validation as well if fp16 is enabled
                                if fp16 and torch.cuda.is_available():
                                    with torch.cuda.amp.autocast():
                                        outputs = model(
                                            input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            labels=labels
                                        )
                                else:
                                    outputs = model(
                                        input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        labels=labels
                                    )
                                
                                loss = outputs.loss

                                # When using DataParallel, make sure loss is a scalar
                                if multi_gpu and torch.cuda.device_count() > 1:
                                    loss = loss.mean()
                                
                                val_step_loss += loss.item()
                                
                                logits = outputs.logits
                                preds = torch.argmax(logits, dim=1).cpu().numpy()
                                val_predictions.extend(preds)
                                val_true_labels.extend(labels.cpu().numpy())
                        
                        avg_val_step_loss = val_step_loss / len(val_loader)
                        val_accuracy = accuracy_score(val_true_labels, val_predictions)
                        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                            val_true_labels, val_predictions, average='binary'
                        )
                        
                        print(f"Step {global_step} validation - Loss: {avg_val_step_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
                        
                        # Save model if it's the best so far
                        if avg_val_step_loss < best_val_loss:
                            best_val_loss = avg_val_step_loss
                            # Save the actual model (unwrap if using DataParallel)
                            save_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                            save_model.save_pretrained(best_model_path)
                            print(f"New best model saved! (Loss: {avg_val_step_loss:.4f}, F1: {val_f1:.4f})")
                        
                        # Return to training mode
                        model.train()
            else:
                # If we're in the middle of accumulation steps, show accumulation progress
                if gradient_accumulation_steps > 1:
                    train_pbar.set_postfix({
                        'loss': loss.item() * gradient_accumulation_steps, 
                        'acc_step': f'{(step % gradient_accumulation_steps) + 1}/{gradient_accumulation_steps}'
                    })
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Average training loss: {avg_train_loss:.4f}")
        
        # Full validation at the end of each epoch
        model.eval()
        val_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]")
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Use autocast for validation as well if fp16 is enabled
                if fp16 and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                loss = outputs.loss

                # When using DataParallel, make sure loss is a scalar
                if multi_gpu and torch.cuda.device_count() > 1:
                    loss = loss.mean()
                
                val_loss += loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({'loss': loss.item()})
        
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
        
        print(f"Epoch {epoch+1}/{epochs} - Full validation results:")
        print(f"  Loss: {avg_val_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        # Save best model at epoch level too
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epoch_checkpoint_path = os.path.join(output_dir, f"best_model_epoch_{epoch+1}")
            # Save the actual model (unwrap if using DataParallel)
            save_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            save_model.save_pretrained(epoch_checkpoint_path)
            
            # Update the main best model
            save_model.save_pretrained(best_model_path)
            print(f"New best model saved! (Loss: {avg_val_loss:.4f}, F1: {f1:.4f})")
    
    # Return the unwrapped model if it was wrapped with DataParallel
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model

def evaluate_model(model, test_loader, device='cuda:0', fp16=False):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Use autocast for evaluation if fp16 is enabled
            if fp16 and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': predictions,
        'true_labels': true_labels
    }

def predict_new_data(model_path, data_df, tokenizer_name="BAAI/bge-m3", max_length=512, batch_size=32, device='cuda:0', fp16=False):
    """
    Predict labels for new data points and fill the label column.
    
    Args:
        model_path (str): Path to the saved model directory
        data_df (pd.DataFrame): DataFrame with columns 'text', 'label' (empty), 'job_id'
        tokenizer_name (str): Name of the tokenizer to use (should match training)
        max_length (int): Maximum sequence length for tokenization
        batch_size (int): Batch size for prediction
        device (str): Device to use for prediction
        fp16 (bool): Whether to use mixed precision for inference
    
    Returns:
        pd.DataFrame: DataFrame with filled label column containing predictions
    """
    print(f"Loading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_name)
    
    # Load trained model
    model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    # Create a copy of the dataframe to avoid modifying the original
    result_df = data_df.copy()
    
    # Extract texts for prediction
    texts = data_df['text'].values
    
    # Create dummy labels (not used for prediction)
    dummy_labels = np.zeros(len(texts))
    
    # Create dataset for prediction
    predict_dataset = TextClassificationDataset(
        texts=texts,
        labels=dummy_labels,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create data loader
    predict_loader = DataLoader(
        predict_dataset,
        batch_size=batch_size,
        shuffle=False  # Important: don't shuffle for prediction to maintain order
    )
    
    # Make predictions
    predictions = []
    print("Making predictions...")
    
    with torch.no_grad():
        for batch in tqdm(predict_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Use autocast for prediction if fp16 is enabled
            if fp16 and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(batch_preds)
    
    # Fill the label column with predictions
    result_df['label'] = predictions
    
    print(f"Completed predictions for {len(predictions)} samples")
    print(f"Prediction distribution: {np.bincount(predictions)}")
    
    return result_df

def predict_from_pickle(model_path, input_pickle_path, output_pickle_path=None, output_csv_path=None, tokenizer_name="BAAI/bge-m3", max_length=512, batch_size=32, device='cuda:0', fp16=False):
    """
    Convenience function to predict from pickle file and save results.
    
    Args:
        model_path (str): Path to the saved model directory
        input_pickle_path (str): Path to input pickle file with DataFrame containing columns 'text', 'label' (empty), 'job_id'
        output_pickle_path (str): Path to save the pickle file with predictions (optional)
        output_csv_path (str): Path to save the CSV with predictions (optional)
        tokenizer_name (str): Name of the tokenizer to use
        max_length (int): Maximum sequence length for tokenization
        batch_size (int): Batch size for prediction
        device (str): Device to use for prediction
        fp16 (bool): Whether to use mixed precision for inference
    
    Returns:
        pd.DataFrame: DataFrame with filled label column containing predictions
    """
    print(f"Loading data from {input_pickle_path}...")
    
    # Load data from pickle file
    with open(input_pickle_path, 'rb') as f:
        data_df = pickle.load(f)
    
    # Validate that it's a DataFrame
    if not isinstance(data_df, pd.DataFrame):
        raise ValueError(f"Pickle file should contain a pandas DataFrame, got {type(data_df)}")
    
    # Validate required columns
    required_columns = ['text', 'label', 'job_id']
    missing_columns = [col for col in required_columns if col not in data_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    print(f"Loaded {len(data_df)} samples")
    
    # Make predictions
    result_df = predict_new_data(
        model_path=model_path,
        data_df=data_df,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        fp16=fp16
    )
    
    # Save results if output paths are provided
    if output_pickle_path:
        with open(output_pickle_path, 'wb') as f:
            pickle.dump(result_df, f)
        print(f"Results saved to pickle file: {output_pickle_path}")
    
    if output_csv_path:
        result_df.to_csv(output_csv_path, index=False)
        print(f"Results saved to CSV file: {output_csv_path}")
    
    return result_df

def predict_from_csv(model_path, input_csv_path, output_csv_path=None, tokenizer_name="BAAI/bge-m3", max_length=512, batch_size=32, device='cuda:0', fp16=False):
    """
    Convenience function to predict from CSV file and save results.
    
    Args:
        model_path (str): Path to the saved model directory
        input_csv_path (str): Path to input CSV file with columns 'text', 'label' (empty), 'job_id'
        output_csv_path (str): Path to save the CSV with predictions (optional)
        tokenizer_name (str): Name of the tokenizer to use
        max_length (int): Maximum sequence length for tokenization
        batch_size (int): Batch size for prediction
        device (str): Device to use for prediction
        fp16 (bool): Whether to use mixed precision for inference
    
    Returns:
        pd.DataFrame: DataFrame with filled label column containing predictions
    """
    print(f"Loading data from {input_csv_path}...")
    
    # Load data
    data_df = pd.read_csv(input_csv_path)
    
    # Validate required columns
    required_columns = ['text', 'label', 'job_id']
    missing_columns = [col for col in required_columns if col not in data_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    print(f"Loaded {len(data_df)} samples")
    
    # Make predictions
    result_df = predict_new_data(
        model_path=model_path,
        data_df=data_df,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        fp16=fp16
    )
    
    # Save results if output path is provided
    if output_csv_path:
        result_df.to_csv(output_csv_path, index=False)
        print(f"Results saved to {output_csv_path}")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description='Train XLM-RoBERTa classifier for job postings')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the training dataset CSV')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test dataset CSV')
    parser.add_argument('--model_name', type=str, default="BAAI/bge-m3", help='Pretrained model name')
    parser.add_argument('--train_batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Evaluation batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Warmup steps')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--output_dir', type=str, default="./models", help='Output directory for models')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to use (cuda:0, cuda:1, cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--text_column', type=str, default="text", help='Column name for text data')
    parser.add_argument('--label_column', type=str, default="label", help='Column name for label data')
    parser.add_argument('--log_steps', type=int, default=100, help='Print loss and validate every n steps')
    # New arguments for multi-GPU and gradient accumulation
    parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs for training if available')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training (FP16)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    
    # Split training data to create validation set
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=args.seed)
    
    # Prepare tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name)
    
    # Create datasets
    train_dataset = TextClassificationDataset(
        texts=train_df[args.text_column].values,
        labels=train_df[args.label_column].values,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    val_dataset = TextClassificationDataset(
        texts=val_df[args.text_column].values,
        labels=val_df[args.label_column].values,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    test_dataset = TextClassificationDataset(
        texts=test_df[args.text_column].values,
        labels=test_df[args.label_column].values,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # Create data loaders with adjusted batch size for multi-GPU if needed
    effective_batch_size = args.train_batch_size
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs")
        # We can keep the same batch size as it will be divided across GPUs
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False
    )
    
    # Print dataset information
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    print(f"Loading model: {args.model_name}")
    # Load model with fp16 if specified
    # if args.fp16 and torch.cuda.is_available():
    #     print("Loading model in half precision (FP16)")
    #     from transformers import AutoConfig
    #     config = AutoConfig.from_pretrained(args.model_name, num_labels=2)
    #     model = XLMRobertaForSequenceClassification.from_pretrained(
    #         args.model_name,
    #         config=config,
    #         torch_dtype=torch.float16
    #     )
    # else:
    model = XLMRobertaForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2
    )
    
    # Set device - if multi_gpu is True, we'll use 'cuda' and let DataParallel handle the rest
    device = args.device
    if args.multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = 'cuda'
    
    model = model.to(device)
    
    # Train model with new parameters
    print("Starting training...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        device=device,
        log_steps=args.log_steps,
        output_dir=args.output_dir,
        multi_gpu=args.multi_gpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    results = evaluate_model(model, test_loader, device=device, fp16=args.fp16)
        
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_classifier")
    model.save_pretrained(final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Print final results
    print("\nFinal Test Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")

if __name__ == "__main__":
    # Check if this is being run for prediction or training
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        # Prediction mode
        predict_parser = argparse.ArgumentParser(description='Make predictions with trained XLM-RoBERTa classifier')
        predict_parser.add_argument('predict', help='Command to run prediction mode')
        predict_parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model directory')
        predict_parser.add_argument('--input_pickle', type=str, help='Path to input pickle file containing DataFrame')
        predict_parser.add_argument('--input_csv', type=str, help='Path to input CSV file (alternative to pickle)')
        predict_parser.add_argument('--output_pickle', type=str, help='Path to save predictions as pickle file')
        predict_parser.add_argument('--output_csv', type=str, help='Path to save predictions as CSV file')
        predict_parser.add_argument('--tokenizer_name', type=str, default="BAAI/bge-m3", help='Tokenizer name (should match training)')
        predict_parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
        predict_parser.add_argument('--batch_size', type=int, default=32, help='Batch size for prediction')
        predict_parser.add_argument('--device', type=str, default="cuda:0", help='Device to use (cuda:0, cuda:1, cpu)')
        predict_parser.add_argument('--fp16', action='store_true', help='Use mixed precision for inference')
        
        predict_args = predict_parser.parse_args()
        
        # Validate input arguments
        if not predict_args.input_pickle and not predict_args.input_csv:
            print("Error: Either --input_pickle or --input_csv must be provided")
            sys.exit(1)
        
        if predict_args.input_pickle and predict_args.input_csv:
            print("Error: Provide either --input_pickle or --input_csv, not both")
            sys.exit(1)
        
        # Run prediction
        try:
            if predict_args.input_pickle:
                # Use pickle input
                result_df = predict_from_pickle(
                    model_path=predict_args.model_path,
                    input_pickle_path=predict_args.input_pickle,
                    output_pickle_path=predict_args.output_pickle,
                    output_csv_path=predict_args.output_csv,
                    tokenizer_name=predict_args.tokenizer_name,
                    max_length=predict_args.max_length,
                    batch_size=predict_args.batch_size,
                    device=predict_args.device,
                    fp16=predict_args.fp16
                )
            else:
                # Use CSV input
                result_df = predict_from_csv(
                    model_path=predict_args.model_path,
                    input_csv_path=predict_args.input_csv,
                    output_csv_path=predict_args.output_csv,
                    tokenizer_name=predict_args.tokenizer_name,
                    max_length=predict_args.max_length,
                    batch_size=predict_args.batch_size,
                    device=predict_args.device,
                    fp16=predict_args.fp16
                )
            
            print("\nPrediction completed successfully!")
            if predict_args.output_pickle:
                print(f"Results saved to pickle file: {predict_args.output_pickle}")
            if predict_args.output_csv:
                print(f"Results saved to CSV file: {predict_args.output_csv}")
            if not predict_args.output_pickle and not predict_args.output_csv:
                print("Results not saved. Use --output_pickle or --output_csv to save predictions.")
                
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            sys.exit(1)
    else:
        # Training mode (original main function)
        # main()
        df = pd.read_pickle("data/test_job_postings_chunks.pkl")  # Load pickle file
        predictions_df = predict_new_data(
            model_path="../models/best_classifier",
            data_df=df,
            device="cuda:2")
        predictions_df.to_pickle("data/test_job_postings_chunks_predictions.pkl")