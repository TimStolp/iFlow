import argparse
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader
import torchvision

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


if __name__ == '__main__':

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
    Z_EST_FOLDER = osp.join('z_est/', 'mnist')

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

    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('../mnist_data',
                                                          download=True,
                                                          train=True,
                                                          transform=torchvision.transforms.Compose([
                                                              torchvision.transforms.ToTensor() # first, convert image to PyTorch tensor
                                                              #torchvision.transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs

                                                          ])),
                                           batch_size=args.batch_size,
                                           shuffle=True)
    args.N = len(train_loader.dataset)
    metadata.update({"n": args.N})
    aux_dim = len(train_loader.dataset.classes)
    print(train_loader.dataset.classes)
    metadata.update({"aux_dim": aux_dim})
    data_dim = len(train_loader.dataset[0][0].flatten())
    metadata.update({"data_dim": data_dim})
    latent_dim = data_dim
    metadata.update({"latent_dim": latent_dim})


    if args.max_iter is None:
        args.max_iter = len(train_loader) * args.epochs

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

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
                                                     factor=args.lr_drop_factor, \
                                                     patience=args.lr_patience, \
                                                     verbose=True) # factor=0.1 and patience=4

    ste = time.time()
    print('setup time: {}s'.format(ste - st))

    # setup loggers
    logger = Logger(logdir=LOG_FOLDER) # 'log/'
    exp_id = logger.get_id() # 1

    tensorboard_run_name = TENSORBOARD_RUN_FOLDER + 'exp' + str(exp_id) + '_'.join(
        map(str, ['', args.batch_size, args.max_iter, args.lr, args.hidden_dim, args.depth, args.anneal]))
    # 'runs/exp1_64_12500_0.001_50_3_False'

    writer = SummaryWriter(logdir=tensorboard_run_name)

    if args.i_what == 'iFlow':
        logger.add('log_normalizer')
        logger.add('neg_log_det')
        logger.add('neg_trace')

    logger.add('loss')
    logger.add('bpd')
    print('Beginning training for exp: {}'.format(exp_id))

    # training loop
    epoch = 0
    model.train()
    while epoch < args.epochs: #args.max_iter:  #12500
        est = time.time()
        for itr, (x, u) in enumerate(train_loader):
            x = x.flatten(start_dim=1)
            u = F.one_hot(u, num_classes=aux_dim).float()
            acc_itr = itr + epoch * len(train_loader)

            # x is of shape [64, 4]
            # u is of shape [64, 10], one-hot coding of 10 classes
            # there's no z, since source is not available

            #it += 1
            #model.anneal(args.N, args.max_iter, it)
            optimizer.zero_grad()

            if args.cuda:
                x = x.cuda(device=device, non_blocking=True)
                u = u.cuda(device=device, non_blocking=True)

            if args.i_what == 'iVAE':
                elbo, z_est = model.elbo(x, u) #elbo is a scalar loss while z_est is of shape [64, 2]
                loss = elbo.mul(-1)

            elif args.i_what == 'iFlow':
                ldj = torch.zeros(x.shape[0], device=device)
                x, ldj = dequant_module(x, ldj, reverse=False)
                (log_normalizer, neg_trace, neg_log_det), z_est = model.neg_log_likelihood(x, u)
                loss = log_normalizer + neg_trace + neg_log_det

            loss.backward()
            optimizer.step()

            logger.update('loss', loss.item())
            bpd = loss_to_bpd(loss.item(), data_dim)
            logger.update('bpd', bpd)

            if args.i_what == 'iFlow':
                logger.update('log_normalizer', log_normalizer.item())
                logger.update('neg_trace', neg_trace.item())
                logger.update('neg_log_det', neg_log_det.item())

            if acc_itr % args.log_freq == 0: # % 25
                logger.log()
                writer.add_scalar('data/loss', logger.get_last('loss'), acc_itr)
                writer.add_scalar('data/bpd', logger.get_last('bpd'), acc_itr)

                if args.i_what == 'iFlow':
                    writer.add_scalar('data/log_normalizer', logger.get_last('log_normalizer'), acc_itr)
                    writer.add_scalar('data/neg_trace', logger.get_last('neg_trace'), acc_itr)
                    writer.add_scalar('data/neg_log_det', logger.get_last('neg_log_det'), acc_itr)

                scheduler.step(logger.get_last('loss'))

            if acc_itr % int(args.max_iter / 5) == 0 and not args.no_log:
                checkpoint(TORCH_CHECKPOINT_FOLDER, \
                           exp_id, \
                           acc_itr, \
                           model, \
                           optimizer, \
                           logger.get_last('loss'), \
                           perf=None)

            """
            if args.i_what == 'iVAE':
                print('----epoch {} iter {}:\tloss: {:.4f};\tperf: {:.4f}'.format(\
                                                                   epoch, \
                                                                   itr, \
                                                                   loss.item(), \
                                                                   perf))
            elif args.i_what == 'iFlow':
                print('----epoch {} iter {}:\tloss: {:.4f} (l1: {:.4f}, l2: {:.4f}, l3: {:.4f});\tperf: {:.4f}'.format(\
                                                                    epoch, \
                                                                    itr, \
                                                                    loss.item(), \
                                                                    log_normalizer.item(), \
                                                                    neg_trace.item(), \
                                                                    neg_log_det.item(), \
                                                                    perf))
            """

        epoch += 1
        eet = time.time()
        if args.i_what == 'iVAE':
            print('epoch {}: {:.4f}s;\tloss: {:.4f};\tbpd: {:.4f}'.format(epoch, \
                                                                   eet-est, \
                                                                   logger.get_last('loss'), \
                                                                   logger.get_last('bpd')))
        elif args.i_what == 'iFlow':
            print('epoch {}: {:.4f}s;\tloss: {:.4f};\tbpd: {:.4f} (l1: {:.4f}, l2: {:.4f}, l3: {:.4f})'.format(\
                                                                    epoch, \
                                                                    eet-est, \
                                                                    logger.get_last('loss'), \
                                                                    logger.get_last('bpd'), \
                                                                    logger.get_last('log_normalizer'), \
                                                                    logger.get_last('neg_trace'), \
                                                                    logger.get_last('neg_log_det')))

    et = time.time()
    print('training time: {}s'.format(et - ste))

    writer.close()
    if not args.no_log:
        logger.add_metadata(**metadata)
        logger.save_to_json()
        logger.save_to_npz()

    print('total time: {}s'.format(et - st))

    ###### Run Test Here
    model.eval()
    # download and transform test dataset
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('../mnist_data',
                                                          download=True,
                                                          train=False,
                                                          transform=torchvision.transforms.Compose([
                                                              torchvision.transforms.ToTensor() # first, convert image to PyTorch tensor
                                                              #torchvision.transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                          ])),
                                           batch_size=args.batch_size,#args.batch_size,
                                           shuffle=True)
    bpd_acc = 0
    z_est_list = []
    for x, u in test_loader:
        x = x.flatten(start_dim=1)
        u = F.one_hot(u, num_classes=aux_dim).float().to(device)
        if args.cuda:
            x = x.cuda(device=device, non_blocking=True)
            u = u.cuda(device=device, non_blocking=True)

        if args.i_what == 'iVAE':
            elbo, z_est_batch = model.elbo(x, u)
            loss = elbo.mul(-1)

        elif args.i_what == 'iFlow':
            ldj = torch.zeros(x.shape[0], device=device)
            x, ldj = dequant_module(x, ldj, reverse=False)
            (log_normalizer, neg_trace, neg_log_det), z_est_batch = model.neg_log_likelihood(x, u)
            loss = log_normalizer + neg_trace + neg_log_det

        z_est_list.append(z_est_batch)
        bpd_acc += loss_to_bpd(loss, data_dim)
    z_est = torch.cat((z_est_list), 0)
    test_bpd = bpd_acc / len(test_loader)
    print(f"Test BPD: {test_bpd}")


    z_est = z_est.cpu().detach().numpy()
    if not osp.exists(Z_EST_FOLDER):
        os.makedirs(Z_EST_FOLDER)
    np.save("{}/z_est_{}.npy".format(Z_EST_FOLDER, args.i_what), z_est)
    if args.i_what == 'iFlow':
        # obtain \lambda for each class
        u = torch.eye(aux_dim).float().to(device)
        nat_params = model.nat_params(u)
        nat_params = nat_params.cpu().detach().numpy()
        np.save("{}/nat_params.npy".format(Z_EST_FOLDER), nat_params)
    #print("z_est.shape ==", z_est.shape)
    print("DONE.")
