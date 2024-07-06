import torch
import numpy as np
import time

from utils.data import generate_pcomp_dataset, data_processing, Bag
from utils.algo import pcomp_with_squared_loss, pcomp_with_doubled_hinge_loss, train_baseline, minimax_basis
from utils.model import LinearModel


def run(x1_train, x2_train, 
        given_y1_train, given_y2_train, 
        real_y1_train, real_y2_train, 
        test_data, test_y, 
        args):

    if args.p < 1.0:
        random_idx = np.random.choice(len(x1_train), int(len(x1_train)*args.p), replace=False)
        x1_train = np.array(x1_train)[random_idx].tolist()
        x2_train = np.array(x2_train)[random_idx].tolist()
        given_y1_train = np.array(given_y1_train)[random_idx].tolist()
        given_y2_train = np.array(given_y2_train)[random_idx].tolist()
        real_y1_train = np.array(real_y1_train)[random_idx].tolist()
        real_y2_train = np.array(real_y2_train)[random_idx].tolist()

    if args.me in ['Pcomp_SQ', 'Pcomp_DH']:
        train_bags_without_label = [B.data() for B in x1_train+x2_train]
        basis = minimax_basis(train_bags_without_label, args.degree)
        x1_train = np.concatenate([basis(x.data()).T for x in x1_train], axis=0)
        x2_train = np.concatenate([basis(x.data()).T for x in x2_train], axis=0)
        test_data = np.concatenate([basis(x.data()).T for x in test_data], axis=0)
    else:
        x1_train = x1_train
        x2_train = x2_train
        test_data = test_data

    start_time = time.time()
    if args.me == 'Pcomp_SQ':
        w, b = pcomp_with_squared_loss(x1_train, x2_train, args.prior, args.reg)
    elif args.me == 'Pcomp_DH':
        w, b = pcomp_with_doubled_hinge_loss(x1_train, x2_train, args.prior, args.reg)
    else:
        device = torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        )
        model = LinearModel(x1_train[0].data().shape[1], 1)
        model, test_acc = train_baseline(x1_train, x2_train, test_data, test_y, args, model, device)
        end_time = time.time()
        return None, test_acc, end_time-start_time
    end_time = time.time()

    train_acc = np.mean(np.sign(np.concatenate([x1_train, x2_train], axis=0).dot(w)+b).flatten() == np.concatenate([real_y1_train, real_y2_train]))
    test_acc = np.mean(np.sign(test_data.dot(w)+b).flatten() == test_y)
    return train_acc, test_acc, end_time-start_time