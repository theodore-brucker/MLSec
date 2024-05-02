from conf import * 

import numpy as np
import pandas as pd
import os 
import collections

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader, Dataset

# Set the base directory relative to the script location
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the data
flows1 = pd.read_csv(benign_path)
flows2 = pd.read_csv(mixed_path, low_memory=False)

flows = pd.concat([flows1[:100000], flows2[:100000]], axis=0, ignore_index=True)

#########################################
# Processing in accordance to the paper #
# Autoencoder based anamoly detection   #
# ieeexplore.ieee.org/document/6519239  #
#########################################

class CICDataset(Dataset):
    def __init__(self, data, indices, labels=None, augment=False):
        self.data = torch.stack([data[i] for i in indices])
        self.labels = torch.tensor([labels[i] for i in indices]) if labels is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        data_item = data_item.unsqueeze(0)  # This adds a channel dimension
        if self.augment and (self.labels is not None and self.labels[idx] == 0):
            data_item = self.augment_data(data_item)
        return data_item, self.labels[idx] if self.labels is not None else data_item

    def augment_data(self, data_item):
        # Noise injection
        noise = torch.randn_like(data_item) * 0.01  # Adjust noise level
        data_item += noise
        # Normalization
        if torch.max(data_item) != torch.min(data_item):
            data_item = (data_item - torch.min(data_item)) / (torch.max(data_item) - torch.min(data_item))
        return data_item
    
def clean_dataframe(df):
    initial_columns_count = df.shape[1]
    initial_rows_count = df.shape[0]

    for col in df.columns:
        if df[col].dtype == 'object':
            # Use errors='coerce' to turn parsing errors into NaNs
            df[col] = pd.to_numeric(df[col], errors='coerce')

    numeric_df = df.select_dtypes(include=[np.number])
    df = numeric_df.loc[:, (numeric_df != 0).any(axis=0)]
    df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]
    df = df[(df >= 0).all(axis=1)]

    df.reset_index(drop=True, inplace=True)

    final_columns_count = df.shape[1]
    final_rows_count = df.shape[0]

    print(f"Initial number of columns: {initial_columns_count}, final number of columns: {final_columns_count}")
    print(f"Initial number of rows: {initial_rows_count}, final number of rows: {final_rows_count}")

    return df

def encode_labels(labels):
    # Converts a list of labels into integer format where each unique label is assigned a unique integer.
    # Additionally, prints the total number of unique labels and the number of occurrences of each label.

    # Create a dictionary to map each label to a unique integer
    unique_labels = sorted(set(labels))
    label_mapping = {label: i for i, label in enumerate(unique_labels)}

    # Count occurrences of each label
    label_counts = collections.Counter(labels)
    
    # Print the total number of unique labels and occurrences of each
    print(f"Total number of unique labels: {len(unique_labels)}")
    print("Occurrences of each label:")
    for label, count in sorted(label_counts.items(), key=lambda x: label_mapping[x[0]]):
        print(f"{label}: {count}")

    # Apply the mapping to the labels list to create the encoded labels list
    encoded_labels = [label_mapping[label] for label in labels]

    return encoded_labels


# Encode and separate the labels from the features
flows['Label'] = encode_labels(flows['Label'])

# Clean the feature space and drop 5 columns to create a 64 feature shape
flows_semi_reduced = clean_dataframe(flows.drop(['Timestamp', 'Active Std', 'Active Max','Active Min',], axis=1))
flow_labels = flows_semi_reduced['Label']
flows_reduced = flows_semi_reduced.drop(['Label'], axis = 1)

########################################################
# How triangle area mapping was presented in the paper #
########################################################

def compute_matrix(vector):
    vector_transformed = np.square(vector)
    matrix = np.outer(vector, vector)
    max_val = np.max(matrix)
    if max_val > 0:
        matrix /= max_val
    return matrix

##############################
# Real triangle area mapping #
##############################

def compute_tam_matrix(vector):
    m = len(vector)  # Number of features in the vector
    TAM = np.zeros((m, m))  # Initialize the TAM matrix
    
    # Compute the triangle areas for each pair (j, k) where j != k
    for j in range(m):
        for k in range(j + 1, m):  # Only consider k > j to ensure j != k
            # Triangle area calculation simplified
            triangle_area = 0.5 * abs(vector[j] * vector[k])
            TAM[j, k] = triangle_area
            TAM[k, j] = triangle_area  # Symmetry of the matrix

    return TAM


def vectors_to_matrices(df):    
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    # Ensure all data is numeric
    if not df.applymap(np.isreal).all().all():
        raise ValueError("All columns must be numeric")

    # Convert all columns to float for consistent processing and handling
    df = df.astype(float)

    # Use vectorized operations where possible
    matrices = [compute_tam_matrix(df.iloc[i].values) for i in range(len(df))]

    return matrices

matrices = vectors_to_matrices(flows_reduced)
flow_tensors = [torch.tensor(matrix, dtype=torch.float32) for matrix in matrices]
train_indices, test_indices = train_test_split(
    np.arange(len(flow_labels)),  # create an index array
    test_size=0.2,  # 20% of the data will be used for testing
    random_state=42,  # seed for reproducibility
    stratify=flow_labels  # preserve class distribution
)

labels_array = np.array(flow_labels, dtype=np.int32)
train_dataset = CICDataset(flow_tensors, train_indices, labels=flow_labels)
test_dataset = CICDataset(flow_tensors, test_indices, labels=flow_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

for data, labels in train_loader:
    input_width = data.shape[3]
    print("Input batch shape from DataLoader:", data.shape)
    print("Labels batch shape from DataLoader:", labels.shape)
    break

def process_CIC_CAE():
    # Change the working directory to the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    train_loader_path = os.path.join(CSE_CIC_path, 'train.pth')
    test_loader_path = os.path.join(CSE_CIC_path, 'test.pth')

    # Save the train and test loaders
    torch.save(train_loader, train_loader_path)
    torch.save(test_loader, test_loader_path)
    
    print(f'Saved dataloaders to {CSE_CIC_path}')
