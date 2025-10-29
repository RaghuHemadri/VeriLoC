import os
import torch
import pickle
import joblib


import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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
                padded_embeddings_list.append(np.array(window))
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

            file_start_index = current_index
            # Iterate through all lines in the embeddings dict
            for line_no, value in embeddings.items():
                line = value['verilog_code_line']
    
                if remove_comments and (len(line.strip().split("//")[0]) > 0):
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

def train_autoencoder(train_embeddings, train_embeddings_tensor, test_embeddings_tensor, device):
    # # Autoencoder hyperparameters
    input_dim = train_embeddings.shape[1]  # Dimensionality of the input embeddings
    latent_dim = 128  # Desired dimensionality after reduction
    batch_size = 128
    epochs = 200
    learning_rate = 0.0001
    weight_decay = 1e-5
    # torch.manual_seed(0)
    lambda_orth = 0.1

    # Initialize the autoencoder and move to GPU
    autoencoder = Autoencoder(input_dim, latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(autoencoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Prepare data loaders
    train_dataset = TensorDataset(train_embeddings_tensor, train_embeddings_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_embeddings_tensor, test_embeddings_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=50
    # )

    autoencoder.apply(weights_init)

    train_losses = []
    test_losses = []
    # Training loop with GPU
    for epoch in range(epochs):
        autoencoder.train()
        epoch_loss = 0
        for batch in train_loader:
            inputs, _ = batch
            inputs = inputs.to(device)  # Ensure inputs are on GPU
            
            latent, reconstructed = autoencoder(inputs)
            reconstruction_loss = criterion(reconstructed, inputs)
            
            # Orthogonal regularization
            # encoder_weights = autoencoder.encoder[0].weight
            # orth_loss = orthogonal_loss(encoder_weights)

            # loss = reconstruction_loss + lambda_orth * orth_loss
            loss = reconstruction_loss

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            epoch_loss += loss.item()
        # print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.6f}", end = "\r")
        train_losses.append(epoch_loss / len(train_loader))

        autoencoder.eval()
        test_loss = 0.0
        with torch.no_grad():  # Disable gradient computation
            for batch in test_loader:
                inputs, _ = batch
                inputs = inputs.to(device)  # Ensure inputs are on the same device as the model
                _, reconstructed = autoencoder(inputs)
                loss = criterion(reconstructed, inputs)
                test_loss += loss.item()
        
        # Average test loss
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {test_loss:.6f}", end = "\r")

    return autoencoder, train_losses, test_losses

def plot_losses(train_losses, test_losses):
    # Plot train and test losses
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(range(len(test_losses)), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Test Losses')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    json_data = get_congestion_timing_data()
    file_embedding_data = load_module_embeddings()

    train_files, test_files = get_train_test_files(json_data)
    use_context = True

    train_embeddings, train_labels, train_row_indices = get_embeddings_labels(train_files, json_data, file_embedding_data, use_context)
    test_embeddings, test_labels, test_row_indices = get_embeddings_labels(test_files, json_data, file_embedding_data, use_context)

    with open('../trained_models/scaler.pkl', 'rb') as f:
        scaler = joblib.load(f)

    # scaler = StandardScaler()
    # train_embeddings = scaler.fit_transform(train_embeddings)
    test_embeddings = scaler.transform(test_embeddings)
    train_embeddings = scaler.transform(train_embeddings)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert embeddings to PyTorch tensors and move to GPU
    train_embeddings_tensor = torch.tensor(train_embeddings, dtype=torch.float32).to(device)
    test_embeddings_tensor = torch.tensor(test_embeddings, dtype=torch.float32).to(device)

    autoencoder, train_losses, test_losses = train_autoencoder(train_embeddings, train_embeddings_tensor, test_embeddings_tensor, device)

    torch.save(autoencoder, '../trained_models/autoencoder.pth')
    # plot_losses(train_losses, test_losses)

if __name__ == "__main__":
    main()
