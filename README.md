Requirements:
Python 3.8.12
cvxopt==1.3.0
numpy==1.22.2
scikit_learn==1.2.1
scipy==1.8.0
torch==1.10.2

DEMO

-ds: specifies the dataset
-me: specifies the method
-lr: learning rate for baselines
-wd: weight decay for baselines
-reg: regularization parameter for our Pcomp_SQ and Pcomp_DH
-prior: specifies class prior
-dgree: degree of polynomial kernel
-p: proportion of data used


Examples:

python main.py -ds musk1 -me Pcomp_SQ -degree 1 -reg 100 -prior 0.5
python main.py -ds musk1 -me Pcomp_DH -degree 1 -reg 1000 -prior 0.5 -p 0.2
python main.py -ds musk1 -me BL_SQ -lr 1e-3 -wd 1e-3 -prior 0.5
python main.py -ds musk1 -me BL_DH -lr 1e-3 -wd 1e-1 -prior 0.5 -p 0.2

Thank you for your time!