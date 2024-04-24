import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PACKET_DIRECTORY = 'C:/Users/theob/Code/COS-475-Project/'

dataframes = []
print("Loading in the csvs")
for filename in os.listdir(PACKET_DIRECTORY):
    if filename.endswith('.csv'):
        file_path = os.path.join(PACKET_DIRECTORY, filename)
        df = pd.read_csv(file_path, low_memory=False)
        dataframes.append(df)
df = pd.concat(dataframes, ignore_index=True)

# Converting protocol column into numeric representation
df_encoded = pd.get_dummies(df, columns=['protocol'], drop_first=True)
# Make sure all other data is numeric and fill missing values
df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)
# Convert to a consistent data type
df_encoded = df_encoded.astype('float32')

# Split the dataset into training and testing sets
print("Splitting the dataset into test train")
X_train, X_test = train_test_split(df_encoded, test_size=0.2, random_state=42)

print("Scaling the features using StandardScaler")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Sigmoid(),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
# Visualization function
def visualize_samples(samples_original, samples_reconstructed, n=10):
    fig, axs = plt.subplots(2, n, figsize=(20, 4))
    for i in range(n):
        axs[0, i].imshow(samples_original[i].reshape(8, 1), cmap='gray')  # Adjust the reshape according to your data shape
        axs[0, i].set_title("Original")
        axs[0, i].axis('off')
        
        axs[1, i].imshow(samples_reconstructed[i].reshape(8, 1), cmap='gray')  # Adjust the reshape according to your data shape
        axs[1, i].set_title("Reconstructed")
        axs[1, i].axis('off')
    
    plt.suptitle("Sample Original and Reconstructed Data Points")
    plt.show()

# Convert pandas DataFrames to PyTorch Tensors and then to DataLoader
print("Convert the train and test into PyTorch tensors")
X_train_tensor = torch.tensor(X_train).float()
X_test_tensor = torch.tensor(X_test).float()

train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
test_dataset = TensorDataset(X_test_tensor, X_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# Initialize the model and move it to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Initialize the model using {device}")

model = Autoencoder(X_train.shape[1]).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    model.train()
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluation
# Calculate reconstruction errors
model.eval()
reconstruction_errors = []
num_samples_to_visualize = 10
sample_indices = np.random.choice(len(test_dataset), num_samples_to_visualize, replace=False)
samples_original = []
samples_reconstructed = []

with torch.no_grad():
    for idx, data in enumerate(test_loader):
        inputs, _ = data
        inputs = inputs.to(device)
        reconstructions = model(inputs)
        if idx in sample_indices:
            samples_original.extend(inputs.cpu().numpy())
            samples_reconstructed.extend(reconstructions.cpu().numpy())

# Visualize the samples
visualize_samples(samples_original, samples_reconstructed, num_samples_to_visualize)