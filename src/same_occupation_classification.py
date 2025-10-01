import pickle
import json

import pandas as pd
from tqdm import tqdm

response_dict = pickle.load(open("../tmp/indeed_same_job_pos_neg_samples_annotated.pkl", "rb"))
df_embeddings = pd.read_pickle("../tmp/indeed_data_embeddings_annotated.pkl")
train_data = pickle.load(open("../tmp/indeed_train_data_embeddings.pkl", "rb"))

# check if the job_title is the same for the train data and the embeddings
df_tmp = df_embeddings.merge(train_data, on='job_id', how='inner')
assert (df_tmp['job_title_x'] == df_tmp['job_title_y']).all()
assert (df_tmp['job_description_x'] == df_tmp['job_description_y']).all()
assert np.allclose((df_tmp['bge_m3_emb'] - train_data['title_description_emb'].apply(lambda x: x[0])).apply(lambda x: np.linalg.norm(x)), 0)


train_job_ids = set(train_data.job_id.values.tolist())

dataset = []
for k, v in tqdm(response_dict.items()):
    response_json = json.loads(v['response'].choices[0].message.content.replace("```json", "").replace("```", ""))
    entry_a = df_embeddings.iloc[k[0]]
    entry_b = df_embeddings.iloc[k[1]]
    job_id_a = entry_a['job_id']
    job_id_b = entry_b['job_id']        

    # only use the train data for training
    if job_id_a in train_job_ids and job_id_b in train_job_ids:
        dataset.append({
            'job_id_a': job_id_a,
            'job_id_b': job_id_b,
            'embedding_a': entry_a['bge_m3_emb'],
            'embedding_b': entry_b['bge_m3_emb'],
            'is_same_occupation': response_json['label'],
            'isic_section_code_a': entry_a['isic_section_code'],
            'isic_section_code_b': entry_b['isic_section_code'],
        })
print("number of samples: ", len(dataset))
pd.DataFrame(dataset).is_same_occupation.value_counts()

import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           precision_recall_curve, roc_curve, classification_report)
import xgboost as xgb
from sklearn.model_selection import train_test_split

import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# Randomize and split the dataset
random.seed(42)
random.shuffle(dataset)
train_portion = 0.9
train_data = dataset[:int(len(dataset) * train_portion)]
test_data = dataset[int(len(dataset) * train_portion):]

# Create feature matrices and target vectors
X_train = np.vstack([np.hstack([row["embedding_a"], row["embedding_b"], row["embedding_a"] - row["embedding_b"], row["embedding_b"] * row["embedding_a"]]) for row in train_data])
y_train = np.array([int(row["is_same_occupation"].strip() == 'SAME_OCCUPATION') for row in train_data])

X_test = np.vstack([np.hstack([row["embedding_a"], row["embedding_b"], row["embedding_a"] - row["embedding_b"], row["embedding_b"] * row["embedding_a"]]) for row in test_data])
y_test = np.array([int(row["is_same_occupation"].strip() == 'SAME_OCCUPATION') for row in test_data])

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Check class distribution
print(f"Training class distribution: {np.bincount(y_train)}")
print(f"Testing class distribution: {np.bincount(y_test)}")

# # Get a sample of confidence scores for later analysis
# train_confidences = np.array([row["confidence"] for row in train_data])
# test_confidences = np.array([row["confidence"] for row in test_data])
# print(f"Average confidence in training: {train_confidences.mean():.4f}")
# print(f"Average confidence in testing: {test_confidences.mean():.4f}")


def evaluate_model(model, X, y, threshold=0.5, model_name="Model"):
    """Evaluate a model and print metrics"""
    start_time = time.time()
    
    if isinstance(model, xgb.Booster):
        # For XGBoost booster objects
        dtest = xgb.DMatrix(X)
        y_proba = model.predict(dtest)
    else:
        # For scikit-learn compatible models
        # y_proba = model.predict_proba(X)[:, 1]
        y_proba = model.predict(X, output_margin=False)
    
    y_pred = (y_proba >= threshold).astype(int)
    
    inference_time = (time.time() - start_time) / len(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_proba)
    
    # Print metrics
    print(f"\n{model_name} Evaluation:")
    print(f"Threshold: {threshold:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Inference time per sample: {inference_time*1000:.4f} ms")
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Different Sector', 'Same Sector'],
               yticklabels=['Different Sector', 'Same Sector'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Different Sector', 'Same Sector']))
    
    # Return metrics for comparison
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_proba': y_proba,
        'y_pred': y_pred,
        'inference_time': inference_time
    }


def find_optimal_threshold(y_true, y_proba, plot=True):
    """Find the optimal threshold based on F1 score"""
    # Calculate precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Calculate F1 score for each threshold
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    
    # Find the threshold with the highest F1 score
    best_idx = np.argmax(f1_scores[:-1])  # exclude the last element which doesn't correspond to a threshold
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"Optimal threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")
    
    if plot:
        # Plot F1 score vs threshold
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, f1_scores[:-1], 'b-', label='F1 Score')
        plt.axvline(x=best_threshold, color='r', linestyle='--', 
                   label=f'Best Threshold: {best_threshold:.4f}')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs. Threshold')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot precision-recall curve
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, 'b-', label='Precision-Recall curve')
        plt.scatter(recall[best_idx], precision[best_idx], color='red', s=100, 
                   label=f'Best Threshold: {best_threshold:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return best_threshold


def plot_feature_importance(model, feature_names=None, top_n=20):
    """Plot feature importance for the gradient boosting model"""
    if isinstance(model, xgb.Booster):
        # For XGBoost booster objects
        importance = model.get_score(importance_type='gain')
        importance = {int(k.replace('f', '')): v for k, v in importance.items()}
        df = pd.DataFrame({'feature': importance.keys(), 'importance': importance.values()})
    else:
        # For scikit-learn compatible models
        # Get feature importance as a dictionary
        importance_dict = model.get_score(importance_type='weight')  # or 'gain', 'cover', 'total_gain', 'total_cover'

        # If you need it as an array in the same order as your features
        importance = np.zeros(len(feature_names))  # feature_names should be your list of feature names
        for feature, score in importance_dict.items():
            idx = feature_names.index(feature)
            importance[idx] = score

        df = pd.DataFrame({'feature': range(len(importance)), 'importance': importance})
    
    # Sort by importance
    df = df.sort_values('importance', ascending=False)
    
    # Limit to top N features
    if top_n is not None:
        df = df.head(top_n)
    
    # Add feature names if provided
    if feature_names is not None:
        df['feature_name'] = df['feature'].apply(lambda x: feature_names[x])
    else:
        df['feature_name'] = df['feature'].apply(lambda x: f"Feature {x}")
    
    # Plot
    plt.figure(figsize=(10, max(6, len(df) * 0.3)))
    sns.barplot(x='importance', y='feature_name', data=df)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    return df


# XGBoost with default parameters
print("\n========== XGBoost with Default Parameters ==========")
start_time = time.time()

# Create dataset for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],    # Track multiple metrics
    'eta': 0.1,                          # Slightly lower learning rate for better convergence
    'max_depth': 8,                       # Deeper trees for complex embedding relationships
    'min_child_weight': 5,                # Helps with class imbalance
    'subsample': 0.8,
    'colsample_bytree': 0.7,              # Reduce feature sampling for high-dim data
    'colsample_bylevel': 0.7,             # Additional column sampling per level
    'scale_pos_weight': 5/3,              # Adjust for class imbalance (neg_count/pos_count)
    'tree_method': 'hist',                # Efficient for large datasets
    'grow_policy': 'lossguide',           # Often better for embedding data
    'max_bin': 256,                       # More bins for histogram method
    'reg_alpha': 0.1,                     # L1 regularization
    'reg_lambda': 1.0,                    # L2 regularization
}

# Training with better configuration
model_xgb = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,                 # Increase max rounds
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=50,             # More patience before stopping
    verbose_eval=25
)

# # Default parameters
# params = {
#     'objective': 'binary:logistic',
#     'eval_metric': 'logloss',
#     'eta': 0.1,
#     'max_depth': 6,
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,
#     'tree_method': 'hist',  # For faster computation
# }

# # Train the model
# model_xgb = xgb.train(
#     params,
#     dtrain,
#     num_boost_round=1000,
#     evals=[(dtrain, 'train'), (dtest, 'test')],
#     early_stopping_rounds=20,
#     verbose_eval=10
# )

training_time = time.time() - start_time
print(f"Training time: {training_time:.2f} seconds")

# Evaluate the model
results_xgb = evaluate_model(model_xgb, X_test, y_test, model_name="XGBoost (Default)")

# # Find the optimal threshold
# best_threshold_xgb = find_optimal_threshold(y_test, results_xgb['y_proba'])

# # Re-evaluate with the best threshold
# results_xgb_tuned = evaluate_model(model_xgb, X_test, y_test, threshold=best_threshold_xgb, 
#                                  model_name="XGBoost (Optimal Threshold)")

# Plot feature importance
plot_feature_importance(model_xgb, top_n=20)



# Save best model
best_model = model_xgb  # Change to the best model
joblib.dump(best_model, '../tmp/xgbt_clustering/best_gradient_boosting_model.pkl')
print("Best model saved to 'best_gradient_boosting_model.pkl'")

# Function to make predictions with the optimal threshold
def predict_with_threshold(model, features, threshold=0.5):
    """Make predictions with the specified threshold"""
    if isinstance(model, xgb.Booster):
        dtest = xgb.DMatrix(features)
        probs = model.predict(dtest)
    else:
        probs = model.predict_proba(features)[:, 1]
    
    return (probs >= threshold).astype(int), probs

print("\nExample usage for making predictions with the optimal threshold:")
print("from joblib import load")
print("model = load('../tmp/xgbt_clustering/best_gradient_boosting_model.pkl')")
print("predictions, probabilities = predict_with_threshold(model, features, threshold=best_threshold_lgb)")