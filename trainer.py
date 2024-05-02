from models.CAE import ConvAutoencoder, train_CAE, device
from models.SAE import SequenceAutoencoder, train_SAE
from conf import *

import torch
from torch import optim, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

import os
import sys 

base_dir = os.path.dirname(os.path.abspath(__file__))

flows_train_path = os.path.join(CSE_CIC_path, 'train.pth')
flows_test_path = os.path.join(CSE_CIC_path, 'test.pth')
train_loader = torch.load(flows_train_path)
test_loader = torch.load(flows_test_path)


def train_CAE_CSE():
    model = ConvAutoencoder().to(device)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    train_CAE(model, train_loader, optimizer, criterion, device)

def train_SAE_CSE():
    
    model = SequenceAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 regularization

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * 15
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=total_steps)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Learning rate decay
    criterion = nn.MSELoss()
    train_SAE(model, train_loader, test_loader, criterion, optimizer, device, epochs=500, scheduler=scheduler)



def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'SAE':
        print("Training Sequence Autoencoder (SAE)")
        train_SAE_CSE()
    else:
        print("Training Convolutional Sequence Autoencoder (CAE)")
        train_CAE_CSE()

if __name__ == '__main__':
    main()