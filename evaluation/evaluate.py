from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import torch

class AnomalyDetection:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device

    def compute_CAE_mse(self, reconstructed, inputs):
        return ((reconstructed - inputs) ** 2).mean(axis=None)  # if you want the mean over all elements

    def compute_SAE_mse(self, reconstructed, inputs):
        return ((reconstructed - inputs) ** 2).mean(dim=[1, 2])

    def accumulate_SAE_errors(self):
        reconstruction_errors = []
        true_labels = []
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                reconstructed = self.model(inputs)
                mse = self.compute_SAE_mse(reconstructed, inputs)
                reconstruction_errors.append(mse.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        reconstruction_errors = np.concatenate(reconstruction_errors)
        return reconstruction_errors, true_labels
    
    def accumulate_CAE_errors(self):
        reconstruction_errors = []
        true_labels = []
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                reconstructed = self.model(inputs)
                mse = self.compute_CAE_mse(reconstructed, inputs)
                reconstruction_errors.append(mse.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        reconstruction_errors = np.concatenate(reconstruction_errors)
        return reconstruction_errors, true_labels

    def find_best_threshold(self, reconstruction_errors, true_labels, thresholds):
        # Ensure the inputs are numpy arrays for element-wise comparisons
        reconstruction_errors = np.array(reconstruction_errors)
        true_labels = np.array(true_labels)
        
        best_accuracy = 0
        best_threshold = 0

        for threshold in thresholds:
            # Predictions are 1 where errors are greater than the threshold, 0 otherwise
            predictions = (reconstruction_errors > threshold).astype(int)
            accuracy = accuracy_score(true_labels, predictions)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
            
            print(f'Threshold: {threshold:.4f}, Accuracy: {accuracy*100:.2f}%')

        return best_threshold, best_accuracy

    def plot_all_classes(self, reconstruction_errors, true_labels, threshold):
        error_dict = {}
        unique_labels = np.unique(true_labels)
        for label in unique_labels:
            error_dict[label] = reconstruction_errors[true_labels == label]

        self.plot_histogram(error_dict, threshold, unique_labels)

    def plot_histogram(self, error_dict, threshold, labels, zoom_percentile=100):
        plt.figure(figsize=(12, 8))
        colors = plt.cm.get_cmap('viridis', len(labels))

        # Clean data: remove any NaN values from error_dict
        for key in error_dict.keys():
            error_dict[key] = error_dict[key][np.isfinite(error_dict[key])]

        all_errors = np.concatenate(list(error_dict.values()))
        if all_errors.size == 0:
            print("No valid error data available to plot.")
            return
        
        upper_limit = max(np.percentile(all_errors, zoom_percentile), threshold)

        for idx, label in enumerate(labels):
            plt.hist(error_dict[label], bins=20, alpha=0.7, label=f'Label {label}', color=colors(idx))
        
        plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.2f}')
        plt.legend()
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.title('Reconstruction Errors by Class')
        plt.xlim(left=0, right=upper_limit)
        plt.show()

    def run_CAE_analysis(self):
        errors, labels = self.accumulate_CAE_errors()
        thresholds = np.linspace(0, 0.5, num=500)  # Example range and granularity
        best_threshold, best_accuracy = self.find_best_threshold(errors, labels, thresholds)
        print(f'Best Threshold: {best_threshold:.4f}, Highest Accuracy: {best_accuracy*100:.2f}%')
        self.plot_all_classes(errors, labels, best_threshold)

    def run_SAE_analysis(self):
        errors, labels = self.accumulate_SAE_errors()
        thresholds = np.linspace(0, 0.5, num=500)  # Example range and granularity
        best_threshold, best_accuracy = self.find_best_threshold(errors, labels, thresholds)
        print(f'Best Threshold: {best_threshold:.4f}, Highest Accuracy: {best_accuracy*100:.2f}%')
        self.plot_all_classes(errors, labels, best_threshold)