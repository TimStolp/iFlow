import numpy as np
import torch
from lib.metrics import mean_corr_coef as mcc
import os.path as osp
from lib.iFlow import *
import argparse
from lib.data import DataLoaderGPU, create_if_not_exist_dataset
import operator
from functools import reduce
import os
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', default=None, help='path to data file in .npz format. (default None)')
parser.add_argument('-x', '--data-args', type=str, default=None,
                    help='argument string to generate a dataset. '
                         'This should be of the form nps_ns_dl_dd_nl_s_p_a_u_n. '
                         'Usage explained in lib.data.create_if_not_exist_dataset. '
                         'This will overwrite the `file` argument if given. (default None). '
                         'In case of this argument and `file` argument being None, a default dataset '
                         'described in data.py will be created.')
parser.add_argument('-z', '--latent-dim', type=int, default=None,
                    help='latent dimension. If None, equals the latent dim of the dataset. (default None)')
parser.add_argument('-b', '--batch-size', type=int, default=64, help='batch size (default 64)')
parser.add_argument('-e', '--epochs', type=int, default=20, help='number of epochs (default 20)')
parser.add_argument('-m', '--max-iter', type=int, default=None, help='max iters, overwrites --epochs')
parser.add_argument('-g', '--hidden-dim', type=int, default=50, help='hidden dim of the networks (default 50)')
parser.add_argument('-d', '--depth', type=int, default=3, help='depth (n_layers) of the networks (default 3)')
parser.add_argument('-l', '--lr', type=float, default=1e-3, help='learning rate (default 1e-3)')
parser.add_argument('-s', '--seed', type=int, default=1, help='random seed (default 1)')
parser.add_argument('-c', '--cuda', action='store_true', default=False, help='train on gpu')
parser.add_argument('-p', '--preload-gpu', action='store_true', default=False, dest='preload',
                    help='preload data on gpu for faster training.')
parser.add_argument('-a', '--anneal', action='store_true', default=False, help='use annealing in learning')
parser.add_argument('-n', '--no-log', action='store_true', default=False, help='run without logging')
parser.add_argument('-q', '--log-freq', type=int, default=25, help='logging frequency (default 25).')

parser.add_argument('-ml', '--model', type=str, help='Experiment folder name containing the model checkpoint.')
parser.add_argument('-i', '--i-what', type=str, default="iFlow")
parser.add_argument('-ft', '--flow_type', type=str, default="RQSplineFlow")
parser.add_argument('-nb', '--num_bins', type=int, default=8)
parser.add_argument('-npa', '--nat_param_act', type=str, default="Sigmoid")
parser.add_argument('-u', '--gpu_id', type=str, default='0')
parser.add_argument('-fl', '--flow_length', type=int, default=10)
parser.add_argument('-lr_df', '--lr_drop_factor', type=float, default=0.5)
parser.add_argument('-lr_pn', '--lr_patience', type=int, default=10)

args = parser.parse_args()

if args.file is None:
    args.file = create_if_not_exist_dataset(root='data/{}/'.format(args.seed),
                                            arg_str=args.data_args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

metadata = vars(args).copy()
metadata.update({"device": device})

train_loader = DataLoaderGPU(args.file, shuffle=True, batch_size=args.batch_size)
metadata.update(train_loader.get_metadata())


print('Loading model', args.model)
model_path = osp.join(args.model, 'ckpt', '1', '1_ckpt_10000.pth')

print('Loading data', args.file)
A = np.load(args.file)

x = A['x']  # of shape
x = torch.from_numpy(x).to(device)
print("x.shape ==", x.shape)

s = A['s']  # of shape
# s = torch.from_numpy(s).to(device)
print("s.shape ==", s.shape)

u = A['u']  # of shape
u = torch.from_numpy(u).to(device)
print("u.shape ==", u.shape)

checkpoint = torch.load(model_path)
model = iFlow(args=metadata).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

total_num_examples = reduce(operator.mul, map(int, args.data_args.split('_')[:2]))
model.set_mask(total_num_examples)

z_est, nat_params = model.inference(x, u)
z_est = z_est.cpu().detach().numpy()

corr_coeffs = correlation_coefficients(s, z_est, method='pearson')

print(corr_coeffs)

if not os.path.exists('results'):
    os.makedirs('results')
with open(osp.join('results', 'corr_coeffs_per_dimension.txt'), 'a+') as f:
    f.write(str(corr_coeffs) + '\n')