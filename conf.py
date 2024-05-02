import os
import sys

# Project base 
base = os.path.dirname(os.path.abspath(__file__))

# Add the base directory to the system path
sys.path.append(base)

# Define other directories
data_dir = os.path.join(base, "data")
CSE_CIC_path = os.path.join(data_dir, 'CSE-CIC/flows')
# Define the paths to the datasets
benign_path = os.path.join(CSE_CIC_path, 'csv', 'Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv')
mixed_path = os.path.join(CSE_CIC_path, 'csv', 'Friday-16-02-2018_TrafficForML_CICFlowMeter.csv')


checkpoint_dir = os.path.join(base, "models", "checkpoints")
