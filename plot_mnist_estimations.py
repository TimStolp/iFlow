import argparse
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt

from lib.data import SyntheticDataset, DataLoaderGPU, create_if_not_exist_dataset, to_one_hot
from lib.metrics import mean_corr_coef as mcc
from lib.models import iVAE
from lib.planar_flow import *
from lib.iFlow import *
from lib.utils import Logger, checkpoint, loss_to_bpd
from lib.dequantization import Dequantization

import os
import os.path as osp
import pdb

import datetime


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

parser.add_argument('-i', '--i-what', type=str, default="iFlow")
parser.add_argument('-ft', '--flow_type', type=str, default="RQNSF_AG")
parser.add_argument('-nb', '--num_bins', type=int, default=8)
parser.add_argument('-npa', '--nat_param_act', type=str, default="Sigmoid")
parser.add_argument('-u', '--gpu_id', type=str, default='0')
parser.add_argument('-fl', '--flow_length', type=int, default=10)
parser.add_argument('-lr_df', '--lr_drop_factor', type=float, default=0.25)
parser.add_argument('-lr_pn', '--lr_patience', type=int, default=10)

args = parser.parse_args()

now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

EXPERIMENT_FOLDER = osp.join('experiments/', now)
LOG_FOLDER = osp.join(EXPERIMENT_FOLDER, 'log/')
TENSORBOARD_RUN_FOLDER = osp.join(EXPERIMENT_FOLDER, 'runs/')
TORCH_CHECKPOINT_FOLDER = osp.join(EXPERIMENT_FOLDER, 'ckpt/')
#Z_EST_FOLDER = osp.join('z_est/', args.data_args.split('_')[5])

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

print(args)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

st = time.time()

#if args.file is None:
#    args.file = create_if_not_exist_dataset(root='data/{}/'.format(args.seed), arg_str=args.data_args)

metadata = vars(args).copy()
del metadata['no_log'], metadata['data_args']

device = torch.device('cuda' if args.cuda else 'cpu')
print('training on {}'.format(torch.cuda.get_device_name(device) if args.cuda else 'cpu'))

# load data
#if not args.preload:
#    dset = SyntheticDataset(args.file, 'cpu') # originally 'cpu' ????
#    train_loader = DataLoader(dset, shuffle=True, batch_size=args.batch_size)
#    data_dim, latent_dim, aux_dim = dset.get_dims()
#    args.N = len(dset)
#    metadata.update(dset.get_metadata())
#else:
#    train_loader = DataLoaderGPU(args.file, shuffle=True, batch_size=args.batch_size)
#    data_dim, latent_dim, aux_dim = train_loader.get_dims()
#    args.N = train_loader.dataset_len
#    metadata.update(train_loader.get_metadata())

test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('../mnist_data',
                                                          download=True,
                                                          train=False,
                                                          transform=torchvision.transforms.Compose([
                                                              torchvision.transforms.ToTensor() # first, convert image to PyTorch tensor
                                                              #torchvision.transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                          ])),
                                           batch_size=args.batch_size,#args.batch_size,
                                           shuffle=True)
args.N = len(test_loader.dataset)
metadata.update({"n": args.N})
aux_dim = len(test_loader.dataset.classes)
metadata.update({"aux_dim": aux_dim})
data_dim = len(test_loader.dataset[0][0].flatten())
metadata.update({"data_dim": data_dim})
latent_dim = data_dim
metadata.update({"latent_dim": latent_dim})


if args.max_iter is None:
    args.max_iter = len(test_loader) * args.epochs

if args.latent_dim is not None:
    latent_dim = args.latent_dim
    metadata.update({"train_latent_dim": latent_dim})

# define model and optimizer
model = None
if args.i_what == 'iVAE':
    model = iVAE(latent_dim, \
             data_dim, \
             aux_dim, \
             n_layers=args.depth, \
             activation='lrelu', \
             device=device, \
             hidden_dim=args.hidden_dim, \
             anneal=args.anneal) # False
elif args.i_what == 'iFlow':
    metadata.update({"device": device})
    model = iFlow(args=metadata).to(device)
    dequant_module = Dequantization(quants=256)

print('Loading model', "2021-01-27_160117")
model_path = osp.join("experiments", "2021-01-27_160117", 'ckpt', '1', '1_ckpt_3752.pth')

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

for x, u in test_loader:
    x_flat = x.flatten(start_dim=1)
    u = F.one_hot(u, num_classes=aux_dim).float().to(device)
    if args.cuda:
        x_flat = x_flat.cuda(device=device, non_blocking=True)
        u = u.cuda(device=device, non_blocking=True)
    z_est, nat_params = model.inference(x_flat, u)
    x_est, determinants = model.nf._transform.inverse(z_est)
    x_est = x_est.cpu().detach().numpy().reshape(64, 1, 28, 28)
    for i in range(64):
        print('original')
        plt.imshow(x[i][0])
        plt.show()
        print('estimation')
        plt.imshow(x_est[i][0])
        plt.show()