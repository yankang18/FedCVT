import os

from dataset.bhi_dataset import BHIDataset2Party
from dataset.ctr_dataset import Avazu2party, Criteo2party
from torch.utils.data import DataLoader

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

class ForeverDataIterator(object):
    """A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


def get_dataset(dataset_name, **args):
    """
    Get datasets for the specified dataset.

    :param dataset_name: the name of the dataset.
    :param args: various arguments according to different scenaros.
    :return: training dataset, testing dataset, input dimension for each party, and the number of classes
    """

    data_dir = args["data_dir"]
    col_names = None

    if dataset_name == "bhi":
        num_classes = 2
        train_dst = BHIDataset2Party(os.path.join(data_dir, "bhi"), "train", 50, 50, 2)
        test_dst = BHIDataset2Party(os.path.join(data_dir, "bhi"), "test", 50, 50, 2)
        input_dims = [50, 50]
    elif dataset_name == "ctr_avazu":
        input_dims = [11, 10]
        num_classes = 2
        sub_data_dir = "avazu"
        data_dir = os.path.join(data_dir, sub_data_dir)
        train_dst = Avazu2party(data_dir, 'Train', 2, 32)
        test_dst = Avazu2party(data_dir, 'Test', 2, 32)
        col_names = train_dst.feature_list
        print("[INFO] avazu col names:{}".format(col_names))
    elif dataset_name == "criteo":
        input_dims = [11, 10]
        num_classes = 2
        sub_data_dir = "criteo"
        data_dir = os.path.join(data_dir, sub_data_dir)
        train_dst = Criteo2party(data_dir, 'Train', 2, 32)
        test_dst = Criteo2party(data_dir, 'Test', 2, 32)
        col_names = train_dst.feature_list
        print("[INFO] criteo col names:{}".format(col_names))
    else:
        raise Exception("Does not support dataset [{}] for now.".format(dataset_name))

    return train_dst, test_dst, test_dst, input_dims, num_classes, col_names


class SubsetSampler(object):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        # for i in torch.randperm(len(self.indices), generator=self.generator):
        #     yield self.indices[i]
        for i in range(len(self.indices)):
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)