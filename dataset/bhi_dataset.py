import os
import random
import shutil

# import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms

random.seed(0)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(.1, 2.)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

augmentation = [
    # transforms.ToPILImage(),
    transforms.RandomResizedCrop(32, scale=(0.3, 1.)),  # (0.2, 1.0)
    transforms.RandomApply([
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)  # not strengthened (0.4,0.4,0.4,0.1)
    ], p=0.8),
    # transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # normalize
]


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class BHIDataset2Party:

    def __init__(self, data_dir, data_type, height, width, k, seed=0):
        self.party_num = 2
        self.shuffle_within_patient = False
        self.x = []  # the datapath of 2 different png files
        self.y = []  # the corresponding label
        self.data_dir = data_dir
        self.height = height
        self.width = width
        self.k = k
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # transforms.Normalize(mean=[0.89156885, 0.89156885, 0.89156885],
            #                      std=[0.18063523, 0.18063523, 0.18063523]),
        ])
        random.seed(seed)
        patients = [item for item in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, item))]
        patients_num = len(patients)
        train_num = int(patients_num * 0.8)
        random.shuffle(patients)

        if data_type.lower() == 'train':
            patients = patients[:train_num]
            exclude = 9833
        else:
            patients = patients[train_num:]
            # patients = patients[:-2000]
            exclude = 2458
        for patient in patients:
            for l in [0, 1]:
                files = [d for d in os.listdir(os.path.join(data_dir, patient, str(l)))]
                if self.shuffle_within_patient:
                    random.shuffle(files)
                file_combi_num = len(files) // self.party_num
                for i in range(file_combi_num):
                    sample = [os.path.join(data_dir, patient, str(l), files[2 * i + j]) for j in range(self.k)]
                    if l == 1 and exclude > 0:
                        exclude -= 1
                        continue
                    self.x.append(sample)
                    self.y.append(l)

        self.x = np.array(self.x)
        self.y = np.array(self.y)

        print("loaded x ,y :", len(self.x), len(self.y))

    def find_class(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def get_data(self):
        return self.x, self.y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, indexx):  # this is single_indexx
        _views = self.x[indexx]
        data = []
        for index in range(self.k):
            img = Image.open(_views[index])
            if self.transform is not None:
                img = self.transform(img)
            data.append(img)
        label = self.y[indexx]

        return data, torch.tensor(label, dtype=torch.int64)


class BHIAugDataset2Party:

    def __init__(self, data_dir, data_type, height, width, k, seed=0):
        self.party_num = 2
        self.shuffle_within_patient = False
        self.x = []  # the datapath of 2 different png files
        self.y = []  # the corresponding label
        self.data_dir = data_dir
        self.height = height
        self.width = width
        self.k = k
        self.transform = TwoCropsTransform(transforms.Compose(augmentation))

        random.seed(seed)
        patients = [item for item in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, item))]
        patients_num = len(patients)
        train_num = int(patients_num * 0.8)
        random.shuffle(patients)

        if data_type.lower() == 'train':
            patients = patients[:train_num]
            exclude = 9833
        else:
            patients = patients[train_num:]
            exclude = 2458
        for patient in patients:
            for l in [0, 1]:
                files = [d for d in os.listdir(os.path.join(data_dir, patient, str(l)))]
                if self.shuffle_within_patient:
                    random.shuffle(files)
                file_combi_num = len(files) // self.party_num
                for i in range(file_combi_num):
                    sample = [os.path.join(data_dir, patient, str(l), files[2 * i + j]) for j in range(self.k)]
                    if l == 1 and exclude > 0:
                        exclude -= 1
                        continue
                    self.x.append(sample)
                    self.y.append(l)

        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def find_class(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __len__(self):
        return len(self.x)

    def __getitem__(self, indexx):  # this is single_indexx
        _views = self.x[indexx]
        data = []
        labels = []
        for index in range(self.k):
            img = Image.open(_views[index])
            if self.transform is not None:
                img = self.transform(img)
            data.append(img)
        labels.append(self.y[indexx])

        return data, np.array(labels).ravel()


class BHIDataset4Party:

    def __init__(self, data_dir, data_type, height, width, k):
        self.x = []  # the datapath of 2 different png files
        self.y = []  # the corresponding label
        self.data_dir = data_dir
        self.height = height
        self.width = width
        self.k = k
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.89156885, 0.89156885, 0.89156885],
            #                     std=[0.18063523, 0.18063523, 0.18063523]),
        ])

        target_idx = [0, 3, 7, 11]  # list(range(1,13))
        angle = '060'

        self.classes, self.class_to_idx = self.find_class(data_dir)
        subfixes = [str(((i - 1) * 30)).zfill(3) + '_' + str(i).zfill(3) for i in range(1, 13)]
        # print(subfixes)
        # subfixes = ['_' + str(i).zfill(3) + '.png' for i in range(1, 13)]
        for label in self.classes:
            all_files = [d for d in os.listdir(os.path.join(data_dir, label, data_type))]
            # all_off_files = ['_'.join(item.split('_')[:-2]) for item in all_files]
            if len(label.split('_')) == 1:
                all_indexes = list(set([item.split('_')[1] for item in all_files]))
            else:
                all_indexes = list(set([item.split('_')[2] for item in all_files]))
            # all_off_files = [item.split('.')[0] for item in all_files if item[-3:] == 'off']
            # all_off_files = sorted(list(set(all_off_files)))
            # print(all_off_files)

            for ind in all_indexes:
                all_views = ['{}_{}_{}_{}.png'.format(label, ind, angle, sg_subfix) for sg_subfix in subfixes]
                all_views = [os.path.join(data_dir, label, data_type, item) for item in all_views]
                for i in range(3):
                    sample = [all_views[j * 3 + i] for j in range(0, k)]
                    # sample = [all_views[j] for j in range(0, k)]
                    # print(sample)
                    self.x.append(sample)
                    self.y.append([self.class_to_idx[label]])

        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def find_class(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __len__(self):
        return len(self.x)

    def __getitem__(self, indexx):  # this is single_indexx
        _views = self.x[indexx]
        data = []
        labels = []
        for index in range(self.k):
            img = Image.open(_views[index])
            if self.transform is not None:
                img = self.transform(img)
            data.append(img)
        labels.append(self.y[indexx])

        return data, np.array(labels).ravel()


# def filter_bhi_dataset(data_dir):
#     target_dir = data_dir + '_filtered'
#     for patient in os.listdir(data_dir):
#         # os.makedirs(os.path.join(target_dir, patient), exist_ok=True)
#         print(patient)
#         for l in ['0', '1']:
#             sub_dir = os.path.join(data_dir, patient, l)
#             target_sub_dir = os.path.join(target_dir, patient, l)
#             os.makedirs(target_sub_dir, exist_ok=True)
#             for im in os.listdir(sub_dir):
#                 img_array = cv2.imread(os.path.join(sub_dir, im))
#                 if img_array.shape[0] == 50 and img_array.shape[1] == 50:
#                     shutil.copy(os.path.join(sub_dir, im), os.path.join(target_sub_dir, im))


def test_dataset_v2():
    DATA_DIR = '../data/bhi/'
    train_dataset = BHIDataset2Party(DATA_DIR, 'train', 50, 50, 2)
    valid_dataset = BHIDataset2Party(DATA_DIR, 'test', 50, 50, 2)
    n_train = len(train_dataset)
    n_valid = len(valid_dataset)
    print(n_train)
    print(n_valid)
    train_indices = list(range(n_train))
    valid_indices = list(range(n_valid))
    print(len(train_indices))
    print(len(valid_indices))

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    # valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=2,
                                               num_workers=0,
                                               pin_memory=False,
                                               sampler=train_sampler)

    # valid_loader = torch.utils.data.DataLoader(valid_dataset,
    #                                            batch_size=2,
    #                                            num_workers=2,
    #                                            pin_memory=True)
    import matplotlib.pyplot as plt
    for i, (x1, y) in enumerate(train_loader):
        print(y)
        print(type(x1), type(x1[0]), type(x1[0][0]))
        print(len(x1), len(x1[0]), x1[0].shape, y.shape)
        # print(x1[0].shape, y.shape)
        for i, item in enumerate(x1):
            for k in range(item.shape[0]):
                print(np.max(np.transpose(item[k].numpy(), (1, 2, 0))),
                      np.min(np.transpose(item[k].numpy(), (1, 2, 0))))
                img = np.transpose(item[k].numpy(), (1, 2, 0))  # .astype(float)/255.0
                plt.imshow(img)
                plt.savefig('test_{}_{}.png'.format(i, k))
        print(len(x1), y.shape)

        break

        # if i>3:
        #    break


if __name__ == "__main__":
    test_dataset_v2()
    # filter_bhi_dataset('E:/dataset/bhi')
