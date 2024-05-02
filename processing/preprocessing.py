import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataHandler:
    def __init__(self, data_dir, dataset_type='flows', benign_filename='', mixed_filename=''):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.getenv('FLOWS_DIRECTORY', data_dir)
        self.dataset_type = dataset_type
        self.benign_filename = benign_filename
        self.mixed_filename = mixed_filename
        self.data_loaders_dir = os.path.join(self.data_dir, self.dataset_type)
        os.makedirs(self.data_loaders_dir, exist_ok=True)
        self.flows = None
        self.flow_tensors = None
        self.flow_labels = None
        self.train_loader = None
        self.test_loader = None

    def load_data(self):
        benign_path = os.path.join(self.data_dir, self.benign_filename)
        mixed_path = os.path.join(self.data_dir, self.mixed_filename)
        
        if not os.path.exists(benign_path) or not os.path.exists(mixed_path):
            raise FileNotFoundError("One or more data files specified do not exist.")
        
        print(f"Loading benign data from: {benign_path}")
        print(f"Loading mixed data from: {mixed_path}")

        benign_data = pd.read_csv(benign_path, low_memory=False)
        mixed_data = pd.read_csv(mixed_path, low_memory=False)

        self.flows = pd.concat([benign_data[:100000], mixed_data[:100000]], axis=0, ignore_index=True)
        self.encode_labels()

    def encode_labels(self):
        labels = self.flows['Label'].tolist()
        unique_labels = sorted(set(labels))
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        self.flows['Label'] = [label_mapping[label] for label in labels]

    def clean_dataframe(self):
        df = self.flows.drop(['Timestamp', 'Active Std', 'Active Max', 'Active Min'], axis=1)
        numeric_df = df.select_dtypes(include=[np.number])
        df = numeric_df.loc[:, (numeric_df != 0).any(axis=0)]
        df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]
        df = df[(df >= 0).all(axis=1)]
        df.reset_index(drop=True, inplace=True)
        self.flows = df

    def prepare_data(self):
        self.flow_labels = self.flows['Label']
        flows_reduced = self.flows.drop(['Label'], axis=1)
        self.flow_tensors = [torch.tensor(row, dtype=torch.float32) for row in flows_reduced.values]
        self.create_dataloaders()

    def create_dataloaders(self):
        labels_array = np.array(self.flow_labels, dtype=np.int32)
        indices = np.arange(len(labels_array))
        train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels_array)

        train_dataset = CICDataset(self.flow_tensors, train_indices, labels=labels_array[train_indices])
        test_dataset = CICDataset(self.flow_tensors, test_indices, labels=labels_array[test_indices])

        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    def save_dataloaders(self):
        train_loader_path = os.path.join(self.data_loaders_dir, 'train.pth')
        test_loader_path = os.path.join(self.data_loaders_dir, 'test.pth')
        
        torch.save(self.train_loader, train_loader_path)
        torch.save(self.test_loader, test_loader_path)
        
        print(f'Saved train loader to {train_loader_path}')
        print(f'Saved test loader to {test_loader_path}')

class CICDataset(Dataset):
    def __init__(self, data, indices, labels=None, augment=False):
        self.data = [data[i] for i in indices]
        self.labels = labels if labels is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx].unsqueeze(0)  # Adding a channel dimension
        label = self.labels[idx] if self.labels is not None else data_item
        if self.augment:
            data_item = self.augment_data(data_item)
        return data_item, label

    def augment_data(self, data_item):
        noise = torch.randn_like(data_item) * 0.01
        data_item += noise
        if torch.max(data_item) != torch.min(data_item):
            data_item = (data_item - torch.min(data_item)) / (torch.max(data_item) - torch.min(data_item))
        return data_item

def Preprocess(data_dir, data_type, benign_filename, mixed_filename):
    handler = DataHandler(data_dir, data_type, benign_filename, mixed_filename)
    handler.load_data()
    handler.clean_dataframe()
    handler.prepare_data()
    handler.save_dataloaders()
