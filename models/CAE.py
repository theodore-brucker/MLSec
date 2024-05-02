import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from dotenv import load_dotenv

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda starting: ",torch.cuda.is_available())
data_dir = os.getenv('FLOWS_DIRECTORY')

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5), padding=1, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5), padding=1, bias=True),
            nn.Sigmoid()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(5, 5), bias=True),
            nn.Sigmoid(),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(5, 5), padding=2, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def train_autoencoder_CAE(model, train_loader, optimizer, criterion, device, epochs=5, patience=2, name="CAE"):
    model.train()
    model.to(device)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        total_loss = 0
        for data in train_loader:
            inputs, _ = data if len(data) == 2 else (data, data)  # Handle data without labels
            inputs = inputs.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
        
        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print('Early stopping triggered')
            break
    torch.save(model, f'{data_dir}/models/{name}.pth')
    print(f'Training complete: saved to {data_dir}/models/{name}.pth')

def test_CAE():
    # Create a sample input tensor of size (batch_size, channels, height, width)
    # Example dimensions: 1 image, 1 channel (e.g., grayscale), 28x28 pixels
    input_tensor = torch.randn(64, 1, 65, 65)
    print("Input shape:", input_tensor.shape)
    
    # Initialize the model
    model = ConvAutoencoder()
    
    # Forward the input tensor through the model
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)