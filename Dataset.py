import os
import numpy as np
import torch
from torch.utils.data import Dataset

class fMRI_Dataset(Dataset):
    def __init__(self, data_dir, split, mode='train'):
        #直接加载numpy格式的数据
        self.data_path = os.path.join(data_dir, f"{mode}_split{split}_data.npy")
        self.label_path = os.path.join(data_dir, f"{mode}_split{split}_label.npy")
        
        self.data = np.load(self.data_path)  # 形状: (N, 1, T, V, 1)
        self.labels = np.load(self.label_path)  # 形状: (N,)
    
    def __getitem__(self, index):
        data = torch.from_numpy(self.data[index]).float()
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return data, label
    
    def __len__(self):
        return len(self.data)
    
    def get_num_class(self):
        return len(np.unique(self.labels))
