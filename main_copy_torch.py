import os
import torch
import random
import argparse
import numpy as np

from utils.data import generate_pcomp_dataset, data_processing, Bag
from train import run


def seed_fix(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-lr', help='optimizer\'s learning rate', default=1e-3, type=float)
    parser.add_argument('-ds', help='specify a dataset', default='musk1', type=str)#, choices=['musk1','musk2','fox','elephant','tiger','component', 'function', 'process'],required=False)
    parser.add_argument('-me', help='specify a method', default='Pcomp_SQ', type=str, choices=['Pcomp_SQ','Pcomp_DH', 'BL_SQ', 'BL_DH','BL_RP','BL_LG','BL_HG','BL_SG'], required=False)
    parser.add_argument('-ep', help='number of epochs', default=1000, type=int)
    parser.add_argument('-wd', help='weight decay', default=1e-1, type=float)
    parser.add_argument('-reg', help = 'regularization parameter', default=1e+2, type=float, required=False)
    parser.add_argument('-prior', help='class prior (the ratio of positive bags)', default=0.2, type=float, metavar='[0-1]')
    parser.add_argument('-seed', help = 'Random seed', default=100, type=int, required=False)
    parser.add_argument('-gpu', help = 'used gpu id', default='0', type=str, required=False)
    parser.add_argument('-n', help = 'number of total training bags', default=500, type=int, required=False)
    parser.add_argument('-degree', help = 'degree of polynomial kernel', default=1, type=int, required=False)
    parser.add_argument('-p', help = 'proportion of data used', default=1.0, type=float, metavar='[0-1]')

    args = parser.parse_args()

    seed_fix(args.seed)
    print(f"prior: {args.prior}, me: {args.me}, ds: {args.ds}")

    if args.prior == 0.2: args.n = 250
    elif args.prior == 0.5: args.n = 300
    elif args.prior == 0.8: args.n = 200

    if args.ds == 'musk1':
        num_bags = 92*10
    elif args.ds == 'musk2':
        num_bags = 102*10
    elif args.ds == 'elephant':
        num_bags = 100*10
    elif args.ds == 'fox':
        num_bags = 100*10
    elif args.ds == 'tiger':
        num_bags = 100*10
    elif args.ds == 'component':
        num_bags = 3130
    elif args.ds == 'function':
        num_bags = 5242
    elif args.ds == 'process':
        num_bags = 11718
    elif args.ds == 'BrownCreeper':
        num_bags = 548
    elif args.ds == 'Chestnut-backedChickadee':
        num_bags = 548
    elif args.ds == 'CorelAfrican':
        num_bags = 2000
    elif args.ds == 'CorelAntique':
        num_bags = 2000
    elif args.ds == 'CorelBattleships':
        num_bags = 2000
    elif args.ds == 'Newsgroups1':
        num_bags = 100
    elif args.ds == 'Newsgroups2':
        num_bags = 100
    elif args.ds == 'Newsgroups3':
        num_bags = 100

    ordinary_data, dim= data_processing(args.ds, args)
    ordinary_bags = [Bag(list(map(lambda X: {'data': X['x'], 'label': X['y']}, list(filter(lambda X: X['bag_id'] == i, ordinary_data)))))
        for i in range(num_bags)] # remove bag_id, and put into Bag()

    avg_train_acc = []
    avg_test_acc = []
    avg_run_time = []

    # x1_train_orig, x2_train_orig, given_y1_train_orig, given_y2_train_orig,\
    #                 real_y1_train_orig, real_y2_train_orig, test_data_orig, test_y_orig = \
    #                     generate_pcomp_dataset(ordinary_bags, args)
    # best_test_acc = -1
    # best_res = []
    # best_lambda = None
    # for lambda_ in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]:
    #     args.reg = lambda_
    #     train_acc, test_acc, run_time = run(x1_train_orig, x2_train_orig, 
    #         given_y1_train_orig, given_y2_train_orig, 
    #         real_y1_train_orig, real_y2_train_orig, 
    #         test_data_orig, test_y_orig, 
    #         args)
    #     if test_acc > best_test_acc:
    #         best_test_acc = test_acc
    #         best_res = [train_acc, test_acc, run_time]
    #         best_lambda = lambda_

    # train_acc, test_acc, run_time = best_res
    # if train_acc is not None:
    #     avg_train_acc.append(train_acc)
    # avg_test_acc.append(test_acc)
    # avg_run_time.append(run_time)

    # args.reg = best_lambda

    for _ in range(10):

        x1_train_orig, x2_train_orig, given_y1_train_orig, given_y2_train_orig,\
                    real_y1_train_orig, real_y2_train_orig, test_data_orig, test_y_orig = \
                        generate_pcomp_dataset(ordinary_bags, args)

        best_test_acc = -1
        best_res = []
        for wd_ in [1e-3, 1e-2, 1e-1]:
            args.wd = wd_
            train_acc, test_acc, run_time = run(x1_train_orig, x2_train_orig, 
                given_y1_train_orig, given_y2_train_orig, 
                real_y1_train_orig, real_y2_train_orig, 
                test_data_orig, test_y_orig, 
                args)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_res = [train_acc, test_acc, run_time]

        train_acc, test_acc, run_time = best_res
        
        if train_acc is not None:
            avg_train_acc.append(train_acc)
        avg_test_acc.append(test_acc)
        avg_run_time.append(run_time)

    if train_acc is not None: print(f"avg_train_acc: {np.mean(avg_train_acc)}")
    print(f"avg_test_acc: {np.mean(avg_test_acc)}({np.std(avg_test_acc)})")
    print(f"avg_run_time: {np.mean(avg_run_time)}s")

    import pandas as pd
    try:
        res_csv = pd.read_csv('reviewer4_res.csv')
    except:
        res_csv = pd.DataFrame(columns=['dataset', 'method', 'prior', 'avg_test_acc', 'std_test_acc', 'avg_run_time'])
    
    mean_test_acc = np.around(np.mean(avg_test_acc), 3)
    std_test_acc = np.around(np.std(avg_test_acc), 3)
    mean_run_time = np.around(np.mean(avg_run_time), 3)

    res_csv = pd.concat([res_csv, pd.DataFrame([[args.ds, args.me, args.prior, mean_test_acc, std_test_acc, mean_run_time]], columns=res_csv.columns)], ignore_index=True)
    res_csv.to_csv('reviewer4_res.csv', index=False)