import numpy as np
import os
import zipfile
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch

object_type = {
    0:'vehicle', 
    1:'pedestrian', 
    2:'motorcyclist', 
    3:'cyclist', 
    4:'bus', 
    5:'static', 
    6:'background', 
    7:'construction', 
    8:'riderless_bicycle', 
    9:'unknown'
    }

class LargeDataset(Dataset):
    def __init__(self, data, labels, mean, std):  # Accepts 4 parameters (not 5)
        self.data = data
        self.labels = labels
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        normalized = (sample - self.mean) / self.std
        
        # Add channel-first conversion if needed
        # if normalized.ndim == 3:  # For (H, W, C) format
        #     normalized = normalized.transpose(2, 0, 1)
            
        return (
            torch.from_numpy(normalized).float(),
            torch.tensor(self.labels[idx]).float()
        )

def getData(path):
    zip_path = './data/cse-251-b-2025.zip'
    extract_to = './data/'

    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    train_file = np.load(path+"/train.npz")
    train_data = train_file['data']
    test_file = np.load(path+"/test_input.npz")
    test_data = test_file['data']
    print(f"Training Data's shape is {train_data.shape} and Test Data's is {train_data.shape}")
    return train_data, test_data



# trainData, testData = getData("data")
# print(trainData[0][0][0])
