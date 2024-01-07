import numpy as np
from torchvision.transforms import Lambda
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

import sys
# sys.path.append('C:/Users/WINTER/Desktop/Python基础/report')

class AirlineData(Dataset):
    """Dataset"""
    def __init__(self, tag='train'):
        self.tag = tag

        if self.tag=='train':
            df = pd.read_csv("./train.csv",dtype=np.float64)
        elif self.tag =='valid':
            df = pd.read_csv("./valid.csv",dtype=np.float64)
        else:
            df = pd.read_csv("./test.csv",dtype=np.float64)
        self.X = df.iloc[:,:-1]
        self.y = df.iloc[:,-1]
        self.num_features = df.shape[1]-1

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        feature = torch.Tensor(self.X.iloc[idx,:].to_numpy())
        price = self.y.iloc[idx]
        return feature, price
