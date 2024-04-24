import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


class SequenceAutoencoder(nn.Module):
    def __init__(self):
        super(SequenceAutoencoder, self).__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=8, kernel_size=2, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2),  # Intelligent pooling
            nn.Dropout(p=0.25)
        )
        self.encoder_lstm = nn.LSTM(input_size=8, hidden_size=8, batch_first=True)
        
        self.decoder_lstm = nn.LSTM(input_size=8, hidden_size=8, batch_first=True)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(in_channels=8, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Dropout(p=0.5)
        )
    
    def forward(self, x):
        x = self.encoder_conv(x)
        
        x = x.permute(0, 2, 1)  # Adjusting for LSTM input
        
        # Assuming you're dealing with LSTM layers here...
        x, (hn, cn) = self.encoder_lstm(x)
        
        x = self.decoder_conv(x.permute(0, 2, 1))  # Adjusting back for ConvTranspose1D
        
        return x
def train_SAE(model, train_loader, val_loader, criterion, optimizer, device, epochs=500, scheduler=None, patience=5):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0  # Initialize patience counter
    
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_train_loss = 0.0
        
        for sequences, _ in train_loader:
            sequences = sequences.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, sequences)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()  # Set the model to evaluation mode
        total_val_loss = 0.0
        with torch.no_grad():  # Disable gradient computation
            for sequences, _ in val_loader:
                sequences = sequences.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, sequences)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
        
        # Checkpoint and early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('Model checkpoint saved')
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1  # Increment patience counter
        
        if patience_counter > patience:
            print(f'Stopping early at epoch {epoch+1}. No improvement in validation loss for {patience} epochs.')
            break
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)  # For ReduceLROnPlateau
            else:
                scheduler.step()  # For other types of schedulers
        # Configuration and Hyperparameters
    

    print('Training complete')


