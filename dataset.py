from cProfile import label
import os
import json

import torch
import math
import random
import numpy as np
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class EmgDataset(Dataset):
    def __init__(self, data_list, label_list, augment=False) -> None:
        super().__init__()
        self.data = data_list
        self.label = label_list
        self.augment = augment
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_array, label = self.data[index], self.label[index]

        # if self.augment:
        #     rand_len = random.randint(20, 80)
        #     if data_array.shape[0] > rand_len:
        #         del_len = data_array.shape[0] - rand_len
        #         for _ in range(del_len):
        #             rand_row = random.randint(0, data_array.shape[0] - 1)
        #             data_array = np.delete(data_array, rand_row, 0)
        #     elif data_array.shape[0] < rand_len:
        #         insert_len = rand_len - data_array.shape[0]
        #         for _ in range(insert_len):
        #             rand_num = random.randint(0, data_array.shape[0] - 1)
        #             rand_row = data_array[rand_num]
        #             data_array = np.insert(data_array, rand_num, rand_row, 0)
        data_array.astype(np.float32)
        data_array = torch.from_numpy(data_array)
        return torch.FloatTensor(data_array), torch.tensor(label)