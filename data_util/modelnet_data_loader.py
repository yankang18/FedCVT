import os

import cv2
import numpy as np
from sklearn.utils import shuffle


def find_class(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def get_two_party_data(data_dir, data_type, k, c=10):
    classes, class_to_idx = find_class(data_dir)
    print("classes", classes)
    print("class_to_idx", class_to_idx)

    x = list()  # the datapath of 2 different png files
    y = list()  # the corresponding label

    mean = 0.89156885
    std = 0.18063523

    subfixes = ['_' + str(i).zfill(3) + '.png' for i in range(1, 13)]
    for label in classes[:c]:
        all_files = [d for d in os.listdir(os.path.join(data_dir, label, data_type))]
        all_off_files = [item.split('.')[0] for item in all_files if item[-3:] == 'off']
        all_off_files = sorted(list(set(all_off_files)))
        print("{0} all_off_files {1}".format(label, len(all_off_files)))

        for single_off_file in all_off_files:
            all_views = [single_off_file + sg_subfix for sg_subfix in subfixes]
            all_views = [os.path.join(data_dir, label, data_type, item) for item in all_views]
            # if data_type == "test":
            #     random.shuffle(all_views)
            for i in range(6):
                sample = [all_views[j + i * 2] for j in range(0, k)]
                # sample = [all_views[j] for j in range(0, k)]
                # print(sample)
                Xa_img = cv2.imread(sample[0])
                Xa_img = cv2.resize(Xa_img, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
                Xa_img = Xa_img / 255

                Xb_img = cv2.imread(sample[1])
                Xb_img = cv2.resize(Xb_img, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
                Xb_img = Xb_img / 255

                x.append([Xa_img, Xb_img])
                one_hot_label = [0] * 10
                one_hot_label[class_to_idx[label]] = 1
                # self.y.append([self.class_to_idx[label]])
                y.append(one_hot_label)

    x = np.array(x)
    y = np.array(y)
    return shuffle(x[:, 0], x[:, 1], y)


if __name__ == '__main__':
    # data_dir = "../../../Data/"
    data_dir = "../../../Data/modelnet40v1png/"

    num_classes = 10
    xa, xb, y = get_two_party_data(data_dir=data_dir, data_type="train", k=2, c=num_classes)

    print("x_0 shape:{0}".format(xa[0].shape))
    for d in xa[0]:
        print(d)

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    imgplot = plt.imshow(xa[0])
    plt.show()
    # print("x_1 shape:{0}, {1}".format(data[1], data[1].shape))
    # print("y shape:{0}, {1}".format(data[2], data[2].shape))


