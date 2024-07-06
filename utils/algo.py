import numpy as np
from cvxopt.solvers import qp, options
from cvxopt import matrix
import torch

options['show_progress'] = False

def pcomp_with_squared_loss(X1, X2, prior, lambda_ = 1):
    n = X1.shape[0] 
    p = X1.shape[1]+1 
    X1 = np.c_[X1, np.ones(n)]
    X2 = np.c_[X2, np.ones(n)]
    A = ((1-prior)/(4*n))*X1.T.dot(X1)+(prior/(4*n))*X2.T.dot(X2) + (lambda_/2)*np.eye(p)
    B = np.ones((1, n)).dot((-(1+prior)/(2*n))*X1 + ((2-prior)/(2*n))*X2)
    param = -0.5*np.linalg.inv(A).dot(B.T)
    w, b = param[:-1], param[-1]
    return w, b

def pcomp_with_doubled_hinge_loss(X1, X2, prior, lambda_ = 1):
    n = X1.shape[0]
    p = X1.shape[1]+1
    X1 = np.c_[X1, np.ones(n)]
    X2 = np.c_[X2, np.ones(n)]

    P = np.concatenate([
        np.concatenate([lambda_*np.eye(p), np.zeros((p, n)), np.zeros((p, n))], axis=1),
        np.zeros((n, p+n+n)),
        np.zeros((n, p+n+n))
    ])
    q = np.concatenate([
        (X2-prior*X1).T.dot(np.ones((n, 1)))/n,
        ((1-prior)/n)*np.ones((n, 1)),
        (prior/n)*np.ones((n, 1))
    ])
    G = np.concatenate([
        np.concatenate([np.zeros((n, p)), -np.eye(n),       np.zeros((n, n))], axis=1),
        np.concatenate([-X1,              -2*np.eye(n),     np.zeros((n, n))], axis=1),
        np.concatenate([-X1,              -np.eye(n),       np.zeros((n, n))], axis=1),
        np.concatenate([np.zeros((n, p)), np.zeros((n, n)), -np.eye(n)      ], axis=1),
        np.concatenate([-X2,              np.zeros((n, n)), -2*np.eye(n)    ], axis=1),
        np.concatenate([-X2,              np.zeros((n, n)), -np.eye(n)      ], axis=1),
    ])
    h = np.concatenate([
        np.zeros((n, 1)),
        -np.ones((n, 1)),
        np.zeros((n, 1)),
        np.zeros((n, 1)),
        -np.ones((n, 1)),
        np.zeros((n, 1)),
    ])
    result = qp(matrix(P), matrix(q), matrix(G), matrix(h))
    param = np.array(result['x'])
    w = param[:p-1]
    b = float(param[p-1:p])
    return w, b

def baseline_acc(model, bags, labels, device):
    acc = 0
    for i, bag in enumerate(bags):
        X = torch.from_numpy(bag.data().astype(np.float32)).to(device)
        pred = torch.sign(torch.max(model(X))).detach().cpu().numpy()
        acc += (pred == labels[i])
    return acc / len(bags)

def train_baseline(X1_train, X2_train, x_test, y_test, args, model, device):
    

    X1 = torch.from_numpy(np.concatenate([x.data() for x in X1_train], dtype=np.float32)).to(device)
    X2 = torch.from_numpy(np.concatenate([x.data() for x in X2_train], dtype=np.float32)).to(device)
    model = model.to(device)

    pi, theta = args.prior, args.prior
    t1 = (pi*theta*theta - theta*theta + theta - 1)/(theta*(theta-1))
    t2 = theta*(pi-1)/(theta-1)
    t3 = (pi-1)/(theta-1)
    t4 = (pi*theta - theta*theta + theta - 1)/(theta*(theta-1))


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.me == "BL_SQ":
        loss_fn = lambda x: 0.25*torch.square(x-1)
    elif args.me == "BL_DH":
        loss_fn = lambda x: torch.max(-x, torch.max(torch.zeros_like(x), (1-x)/2))
    elif args.me == "BL_HG":
        loss_fn = lambda x: torch.max(torch.zeros_like(x), 1-x)
    elif args.me == "BL_LG":
        loss_fn = lambda x: -torch.nn.functional.logsigmoid(x) #-torch.log(1+torch.exp(-x))
    elif args.me == "BL_RP":
        loss_fn = lambda x: torch.max(torch.zeros_like(x), torch.min(2*torch.ones_like(x), 1-x))/2
    elif args.me == "BL_SG":
        loss_fn = lambda x: torch.sigmoid(-x)

    test_acc_list = []
    for epoch in range(args.ep):
        optimizer.zero_grad()
        y1_hat, y2_hat = model(X1), model(X2)
        loss = (t1*loss_fn(y1_hat) - t2*loss_fn(-y1_hat)).mean() + \
                (t3*loss_fn(-y2_hat) - t4*loss_fn(y2_hat)).mean()
        loss.backward()
        optimizer.step()

        if epoch > args.ep - 10:
            model.eval()
            with torch.no_grad():
                test_acc_list.append(baseline_acc(model, x_test, y_test, device))
            model.train()

    return model, np.mean(test_acc_list)

def minimax_basis(train_bags_without_label, degree=1):
    degree = int(degree)
    stat = lambda X: np.r_[X.min(axis=0), X.max(axis=0)]
    stat_Y_list = np.array([stat(B) for B in train_bags_without_label])
    return lambda X: np.power(stat_Y_list.dot(stat(X).reshape(-1, 1)) + 1, degree)