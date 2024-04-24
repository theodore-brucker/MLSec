import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_reconstructions(model, test_loader, device='cuda', num_samples=5):
    model.eval()  # Set the model to evaluation mode
    sample_inputs, sample_recons = [], []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            if batch_idx >= num_samples:  # Only store num_samples of samples
                break
            inputs = inputs.to(device)
            reconstructed = model(inputs)  # Assuming model outputs (reconstruction)
            sample_inputs.append(inputs.cpu())
            sample_recons.append(reconstructed.cpu())
    
    # Now, visualize the samples and their reconstructions
    for index in range(num_samples):
        plt.figure(figsize=(10, 4))

        # Original Images
        plt.subplot(1, 2, 1)
        plt.title('Original Images')
        original_images = make_grid(sample_inputs[index], nrow=5, padding=2, normalize=True)
        plt.imshow(original_images.permute(1, 2, 0))
        plt.axis('off')

        # Reconstructed Images
        plt.subplot(1, 2, 2)
        plt.title('Reconstructed Images')
        recon_images = make_grid(sample_recons[index], nrow=5, padding=2, normalize=True)
        plt.imshow(recon_images.permute(1, 2, 0))
        plt.axis('off')

        plt.show()