import numpy as np

from dataset.nuswide_data_util_v2 import load_two_party_data
from torchvision import transforms
import torch


class NUSWIDEDataset2Party(object):

    def __init__(self, data_dir, selected_labels, data_type, k=2, neg_label=0, n_samples=-1):
        self.data_dir = data_dir
        self.selected_labels = selected_labels
        self.neg_label = neg_label
        self.n_samples = n_samples
        self.k = k
        [Xa, Xb, y] = load_two_party_data(data_dir, selected_labels, data_type, neg_label, n_samples)
        self.x = [Xa, Xb]
        self.y = y
        # self.transform = transforms.Compose([
        #     transforms.ToTensor()
        # ])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, indexx):  # this is single_indexx
        data = []
        # labels = []
        for i in range(2):
            data_tensor = torch.tensor(self.x[i][indexx], dtype=torch.float32)
            data.append(data_tensor)
        # labels.append(self.y[indexx])

        # print("labels", labels)
        # yy = np.array(labels).ravel()
        # print("yy", yy.shape)
        return data, np.array(self.y[indexx])
        # return data, np.array(labels)
