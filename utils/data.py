import copy
import math
import random
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import sys
from convert_to_pairwise_bags import build_data


# pca
from sklearn.decomposition import PCA

cur_path = sys.path[0]


class Bag(object):
    def __init__(self, instances):
        self.instances = instances
        self.y = (
            min(1, len(
                list(filter(lambda x: x["label"] == 1, instances)))) * 2 - 1
        )  # create bag label
        self.unlabeled = False

    def __repr__(self):
        return "<Bag #data:{}, label:{}>".format(len(self.instances), self.y)

    def data(self):
        return np.array(list(map(lambda x: x["data"], self.instances)))

    def label(self):
        if self.unlabeled:
            return 0
        else:
            return self.y

    def mask(self):
        self.unlabeled = True

    def add_noise(self):
        m = np.max(list(map(lambda x: x["data"], self.instances)), axis=0)
        for i in range(len(self.instances)):
            z = np.random.normal(0, 0.01, m.shape[0])
            w = np.apply_along_axis(
                lambda x: 0 if (x == 0).all(
                ) else 1, 0, self.instances[i]["data"]
            )
            self.instances[i]["data"] += m * z * w

    def pca_reduction(self, pca):
        for i in range(len(self.instances)):
            self.instances[i]["data"] = pca.transform(
                self.instances[i]["data"].reshape(1, -1)
            )[0]

def augment(bags, ratio, add_noise=True):
  # augment the dataset by adding random bags
  N = len(bags)
  for i in np.random.choice(range(N), int(N * (ratio - 1))):
    aug = copy.deepcopy(bags[i])
    if add_noise:
        aug.add_noise()
    bags.append(aug)

  random.shuffle(bags)
  return bags

def extract_bags(bags, Y, with_label=False):
    if with_label:
        return list(filter(lambda B: B.label() == Y, bags))
    else:
        return list(
            map(lambda B: B.data(), list(filter(lambda B: B.label() == Y, bags)))
        )


def instance_level_train_set(bags1, bags2):
    for i in range(len(bags1)):
        if i == 0:
            data_1 = bags1[i].data()
        else:
            data_1 = np.r_[data_1, bags1[i].data()]
    for j in range(len(bags2)):
        if j == 0:
            data_2 = bags2[j].data()
        else:
            data_2 = np.r_[data_2, bags2[j].data()]
    labels_2 = -np.ones((len(data_2), 1))
    labels_1 = np.ones((len(data_1), 1))
    train_data = np.r_[data_1, data_2]
    train_labels = np.r_[labels_1, labels_2]
    train_data = torch.from_numpy(train_data).float()
    train_labels = torch.from_numpy(train_labels).float()

    return train_data, train_labels


def file_processing(data_file, dim, num_bags):
    ordinary_data = []
    with open(data_file) as f:
        for l in f.readlines():
            if l[0] == "#":
                continue
            ss = l.strip().split(
                " "
            )  # ss = ['4776:919:-1', '0:-1.7513811627811062', ...]
            # where 4776 means the 4776-th instance, 919 means the 919-th bag, -1 means the label of the instance
            x = np.zeros(dim)
            for s in ss[1:]:
                # each index and feature for each instance
                i, xi = s.split(":")
                i = int(i) - 1
                xi = float(xi)  # each feature value
                x[int(i)] = xi
            _, bag_id, y = ss[0].split(":")  # get bid_id and label
            ordinary_data.append(
                {"x": x, "y": int(y), "bag_id": int(bag_id)}
            )  # 4777 datum: x, y, bag_id
    return ordinary_data


def load_mat(dataset_name):
    ordinary_set = sio.loadmat(
        cur_path + "\\" + "datasets\\" + dataset_name + ".mat")
    data = ordinary_set["data"]
    labels = ordinary_set["labels"]
    bag_id = ordinary_set["bags_id"]
    ordinary_data = []
    dim = data.shape[1]
    for i in range(len(bag_id)):
        x = np.array(data[i])
        y = labels[i]
        index = bag_id[i]
        ordinary_data.append({"x": x, "y": int(y), "bag_id": int(index)})
    return ordinary_data, dim


def data_processing(dataset_name, args):
    if dataset_name == "musk1":
        ordinary_data = file_processing(
            cur_path + "\\" "datasets\\musk1.data", 166, 92 * 10
        )  # 920 bags?
        dim = 166
    elif dataset_name == "musk2":
        ordinary_data = file_processing(
            cur_path + "\\" "datasets\\musk2.data", 166, 102 * 10
        )
        dim = 166
    elif dataset_name == "elephant":
        ordinary_data = file_processing(
            cur_path + "\\" "datasets\\elephant.data", 230, 200 * 10
        )
        dim = 230
    elif dataset_name == "fox":
        ordinary_data = file_processing(
            cur_path + "\\" "datasets\\fox.data", 230, 200 * 10
        )
        dim = 230
    elif dataset_name == "tiger":
        ordinary_data = file_processing(
            cur_path + "\\" "datasets\\tiger.data", 230, 200 * 10
        )
        dim = 230
    elif (
        dataset_name == "component"
        or dataset_name == "function"
        or dataset_name == "process"
    ):
        ordinary_data, dim = load_mat(dataset_name)
    return ordinary_data, dim

def train_bags_augment(train_bags, dataset_name):
    if dataset_name == "musk1":
        train_bags = augment(train_bags, ratio = 5)
        pass
    elif dataset_name == "musk2":
        positive_train_data = extract_bags(train_bags, 1, with_label=True)
        negative_train_data = extract_bags(train_bags, -1, with_label=True)
        positive_train_data = augment(positive_train_data, ratio = 16)
        negative_train_data = augment(negative_train_data, ratio = 10)
        train_bags = positive_train_data + negative_train_data
    elif dataset_name in ["elephant", "fox", "tiger"]:
        train_bags = augment(train_bags, ratio = 5)
    elif dataset_name == "component":
        positive_train_data = extract_bags(train_bags, 1, with_label=True)
        negative_train_data = extract_bags(train_bags, -1, with_label=True)
        positive_train_data = augment(positive_train_data, ratio = 20)
        negative_train_data = augment(negative_train_data, ratio = 5)
        train_bags = positive_train_data + negative_train_data
    elif dataset_name == "function":
        positive_train_data = extract_bags(train_bags, 1, with_label=True)
        negative_train_data = extract_bags(train_bags, -1, with_label=True)
        positive_train_data = augment(positive_train_data, ratio = 20)
        negative_train_data = augment(negative_train_data, ratio = 2)
        train_bags = positive_train_data + negative_train_data
    elif dataset_name == "process":
        positive_train_data = extract_bags(train_bags, 1, with_label=True)
        negative_train_data = extract_bags(train_bags, -1, with_label=True)
        positive_train_data = augment(positive_train_data, ratio = 12)
        train_bags = positive_train_data + negative_train_data
    return train_bags


def split_train_test(positive_data, negative_data, test_split=0.2, random_state=None):
    # 固定随机种子
    if random_state is not None:
        np.random.seed(random_state)
    np.random.shuffle(positive_data)
    np.random.shuffle(negative_data)
    train_split = 1-test_split
    train_pos_num, train_neg_num = int(len(positive_data)*train_split), int(len(negative_data)*train_split)
    train_data = positive_data[:train_pos_num].tolist() + negative_data[:train_neg_num].tolist()
    test_data = positive_data[train_pos_num:].tolist() + negative_data[train_neg_num:].tolist()
    return train_data, test_data


def generate_pcomp_dataset(ordinary_bags, prior, n):
    positive_data = np.array(extract_bags(ordinary_bags, 1, with_label=True))
    negative_data = np.array(extract_bags(ordinary_bags, -1, with_label=True))

    np.random.shuffle(positive_data)
    np.random.shuffle(negative_data)

    data_pos = extract_bags(positive_data, 1, with_label=True)
    data_neg = extract_bags(negative_data, -1, with_label=True)
    x1_train, x2_train, given_y1_train, given_y2_train, real_y1_train, real_y2_train = build_data(data_pos, data_neg, prior, n)

    test_data_pos = data_pos[len(x1_train):]
    test_data_neg = data_neg[len(x2_train):]

    n_test_min, n_test_max = 1, len(test_data_pos)+len(test_data_neg)
    while n_test_min < n_test_max:
        mid_ = (n_test_min + n_test_max) // 2
        try:
            x1_test, x2_test, _, _, real_y1_test, real_y2_test = build_data(test_data_pos, test_data_neg, prior, mid_)
        except:
            n_test_max = mid_
        else:
            n_test_min = mid_ + 1

    test_data, test_y = x1_test + x2_test, np.concatenate([real_y1_test, real_y2_test])
        
    return x1_train, x2_train, given_y1_train, given_y2_train, real_y1_train, real_y2_train, test_data, test_y