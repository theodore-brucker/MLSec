import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from dotenv import load_dotenv
import os
import sys

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
models_dir = os.path.join(parent_dir, 'models')
sys.path.append(models_dir)

from CAE import ConvAutoencoder, train_autoencoder_CAE, device
from SAE import SequenceAutoencoder, train_autoencoder_SAE

load_dotenv()

def train_CAE(train_loader):
    model = ConvAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    train_autoencoder_CAE(model, train_loader, optimizer, criterion, device)

def train_SAE(train_loader, test_loader):
    model = SequenceAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.MSELoss()
    train_autoencoder_SAE(model, train_loader, test_loader, criterion, optimizer, device, epochs=500, scheduler=scheduler)

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'SAE':
        print("Training Sequence Autoencoder (SAE)")
        sequence_dir = os.getenv('SEQUENCE_DIRECTORY', 'default_sequence_directory_path')
        train_loader = torch.load(os.path.join(sequence_dir, 'train.pth'))
        test_loader = torch.load(os.path.join(sequence_dir, 'test.pth'))
        train_SAE(train_loader, test_loader)
    else:
        flow_dir = os.getenv('FLOWS_DIRECTORY')
        if not os.path.exists(flow_dir):
            raise ValueError(f"The specified directory {flow_dir} does not exist.")

        print("Training Convolutional Sequence Autoencoder (CAE)")
        train_loader = torch.load(os.path.join(flow_dir, 'train.pth'))
        train_CAE(train_loader)

if __name__ == '__main__':
    main()
