import os
import torch
import pickle
import joblib
import sys

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from utils import *
from Autoencoder import Autoencoder, weights_init

def get_windowed_data(embeddings_array, windowed, window_size, row_indices):
    if windowed:
        padded_embeddings_list = []
        for file, (start, end) in row_indices.items():
            for i in range(start, end):
                window = []
                for j in range(-window_size, window_size + 1):
                    if start <= i + j < end:
                        window.append(embeddings_array[i + j])
                    else:
                        window.append(np.zeros_like(embeddings_array[0]))
                padded_embeddings_list.append(np.concatenate(window))
        embeddings_array = np.array(padded_embeddings_list)
    return embeddings_array

def get_embeddings_labels(files, json_data, file_embedding_data, use_context=True):
    embeddings_list = []
    labels = []
    file_row_indices = {}
    current_index = 0
    remove_comments = True
    window = 5

    for file in files:
        embeddings_file_path = json_data[file]['embeddings_file_path']
        verilog_file_name = file
        if use_context:
            if verilog_file_name in file_embedding_data:
                file_embeddings = file_embedding_data[verilog_file_name][0]
            else:
                file_embeddings = np.zeros((5120,))
        if os.path.exists(embeddings_file_path):
            with open(embeddings_file_path, 'rb') as f:
                embeddings = pickle.load(f)
        else:
            embeddings = None
        
        if embeddings is not None:
            num_lines = max(list(embeddings.keys()))
            congestion_data = json_data[file]['congestion_data']
            valid_lines = set()
            if num_lines > 10000:
                for key in congestion_data:
                    for i in range(key - window, key + window + 1):
                        valid_lines.add(int(i))
            else:
                valid_lines = set(range(num_lines+1))

            # Record the start index for the current file

            file_start_index = current_index
            # Iterate through all lines in the embeddings dict
            for line_no, value in embeddings.items():
                line = value['verilog_code_line']
    
                if remove_comments and (len(line.strip().split("//")[0]) > 0) and (int(line_no+1) in valid_lines):
                    embedding = value['embedding']
                    if use_context:
                        embedding = np.concatenate((embedding, file_embeddings))
                    embeddings_list.append(embedding)
                
                    if (line_no+1) in congestion_data:
                        labels.append(1)  # Mark as anomaly (red)
                    else:
                        labels.append(0)  # Mark as non-anomaly (blue)
                
                    current_index += 1
            file_row_indices[file] = (file_start_index, current_index)
                    
    embeddings_array = np.array(embeddings_list)
    labels_array = np.array(labels)

    return embeddings_array, labels_array, file_row_indices

def main():
    json_data = get_congestion_timing_data()
    file_embedding_data = load_module_embeddings()

    train_files, test_files = get_train_test_files(json_data)
    use_context = True

    train_embeddings, train_labels, train_row_indices = get_embeddings_labels(train_files, json_data, file_embedding_data, use_context)
    test_embeddings, test_labels, test_row_indices = get_embeddings_labels(test_files, json_data, file_embedding_data, use_context)

    with open('../trained_models/scaler.pkl', 'rb') as f:
        scaler = joblib.load(f)

    test_embeddings = scaler.transform(test_embeddings)
    train_embeddings = scaler.transform(train_embeddings)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert embeddings to PyTorch tensors and move to GPU
    train_embeddings_tensor = torch.tensor(train_embeddings, dtype=torch.float32).to(device)
    test_embeddings_tensor = torch.tensor(test_embeddings, dtype=torch.float32).to(device)

    autoencoder = torch.load('../trained_models/autoencoder.pth')
    autoencoder.to(device)
    
    autoencoder.eval()
    with torch.no_grad(): 
        train_embeddings_reduced = autoencoder.encoder(train_embeddings_tensor).cpu().numpy()  # Move to CPU for numpy
        test_embeddings_reduced = autoencoder.encoder(test_embeddings_tensor).cpu().numpy()

    X_train = train_embeddings_reduced
    y_train = train_labels
    
    X_test = test_embeddings_reduced
    y_test = test_labels

    windowed = True
    window_size = 3

    # Apply windowing to the training and test data
    X_train = get_windowed_data(X_train, windowed, window_size, train_row_indices)
    X_test = get_windowed_data(X_test, windowed, window_size, test_row_indices)

    # Compute scale_pos_weight (ratio of majority class to minority class)
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    # Train an XGBoost model
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=30,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    # Evaluate the XGBoost model
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non Anomolous", "Anomolous"]))

    print("\nROC AUC Score:")
    print(roc_auc_score(y_test, y_prob))

    # Create a LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)

    # Set LightGBM parameters
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'is_unbalance': True,  # Handles imbalance
        'num_leaves': 100,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'verbose': -1
    }

    # Train LightGBM model
    lgb_model = lgb.train(lgb_params, train_data, num_boost_round=500)

    # Predict and evaluate
    y_prob = lgb_model.predict(X_test)
    y_pred = (y_prob > 0.5).astype(int)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non Anomolous", "Anomolous"]))

    print("\nROC AUC Score:")
    print(roc_auc_score(y_test, y_prob))

    # Save the XGBoost model
    joblib.dump(xgb_model, '../trained_models/xgb_model_congestion.pkl')

    # Save the LightGBM model
    lgb_model.save_model('../trained_models/lgb_model_congestion.txt')


if __name__ == "__main__":
    main()
