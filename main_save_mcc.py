import argparse
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader

from lib.data import SyntheticDataset, DataLoaderGPU, create_if_not_exist_dataset
from lib.metrics import mean_corr_coef as mcc
from lib.models import iVAE
from lib.planar_flow import *
from lib.iFlow import *
from lib.utils import Logger, checkpoint
from scipy.optimize import linear_sum_assignment
import operator
from functools import reduce

import os
import os.path as osp
import pdb

import datetime

def correlation_coefficients(x, y, method='pearson'):
    """
    A numpy implementation of the mean correlation coefficient metric.

    :param x: numpy.ndarray
    :param y: numpy.ndarray
    :param method: str, optional
            The method used to compute the correlation coefficients.
                The options are 'pearson' and 'spearman'
                'pearson':
                    use Pearson's correlation coefficient
                'spearman':
                    use Spearman's nonparametric rank correlation coefficient
    :return: float
    """
    d = x.shape[1]
    if method == 'pearson':
        cc = np.corrcoef(x, y, rowvar=False)[:d, d:]
    elif method == 'spearman':
        cc = spearmanr(x, y)[0][:d, d:]
    else:
        raise ValueError('not a valid method: {}'.format(method))
    cc_matrix = np.abs(cc)

    corr_coefs = cc_matrix[linear_sum_assignment(-1 * cc_matrix)] #.mean()
    return corr_coefs, cc_matrix

def train_model(args, metadata, device='cuda'):
    print('training on {}'.format(torch.cuda.get_device_name(device) if args.cuda else 'cpu'))

    # load data
    if not args.preload:
        dset = SyntheticDataset(args.file, 'cpu')  # originally 'cpu' ????
        train_loader = DataLoader(dset, shuffle=True, batch_size=args.batch_size)
        data_dim, latent_dim, aux_dim = dset.get_dims()
        args.N = len(dset)
        metadata.update(dset.get_metadata())
    else:
        train_loader = DataLoaderGPU(args.file, shuffle=True, batch_size=args.batch_size)
        data_dim, latent_dim, aux_dim = train_loader.get_dims()
        args.N = train_loader.dataset_len
        metadata.update(train_loader.get_metadata())

    if args.max_iter is None:
        args.max_iter = len(train_loader) * args.epochs

    if args.latent_dim is not None:
        latent_dim = args.latent_dim
        metadata.update({"train_latent_dim": latent_dim})

    # define model and optimizer
    model = None
    if args.i_what == 'iVAE':
        model = iVAE(latent_dim,
                     data_dim,
                     aux_dim,
                     n_layers=args.depth,
                     activation='lrelu',
                     device=device,
                     hidden_dim=args.hidden_dim,
                     anneal=args.anneal,  # False
                     file=metadata['file'],  # Added dataset location for easier checkpoint loading
                     seed=1,
                     epochs=args.epochs)
    elif args.i_what == 'iFlow':
        metadata.update({"device": device})
        model = iFlow(args=metadata).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
                                                     factor=args.lr_drop_factor, \
                                                     patience=args.lr_patience, \
                                                     verbose=True)  # factor=0.1 and patience=4

    ste = time.time()
    print('setup time: {}s'.format(ste - st))

    # setup loggers
    logger = Logger(logdir=LOG_FOLDER)  # 'log/'
    exp_id = logger.get_id()  # 1

    tensorboard_run_name = TENSORBOARD_RUN_FOLDER + 'exp' + str(exp_id) + '_'.join(
        map(str, ['', args.batch_size, args.max_iter, args.lr, args.hidden_dim, args.depth, args.anneal]))
    # 'runs/exp1_64_12500_0.001_50_3_False'

    writer = SummaryWriter(logdir=tensorboard_run_name)

    if args.i_what == 'iFlow':
        logger.add('log_normalizer')
        logger.add('neg_log_det')
        logger.add('neg_trace')

    logger.add('loss')
    logger.add('perf')
    print('Beginning training for exp: {}'.format(exp_id))

    # training loop
    epoch = 0
    model.train()
    while epoch < args.epochs:  # args.max_iter:  #12500
        est = time.time()
        for itr, (x, u, z) in enumerate(train_loader):
            acc_itr = itr + epoch * len(train_loader)

            # x is of shape [64, 4]
            # u is of shape [64, 40], one-hot coding of 40 classes
            # z is of shape [64, 2]

            # it += 1
            # model.anneal(args.N, args.max_iter, it)
            optimizer.zero_grad()

            if args.cuda and not args.preload:
                x = x.cuda(device=device, non_blocking=True)
                u = u.cuda(device=device, non_blocking=True)

            if args.i_what == 'iVAE':
                elbo, z_est = model.elbo(x, u)  # elbo is a scalar loss while z_est is of shape [64, 2]
                loss = elbo.mul(-1)

            elif args.i_what == 'iFlow':
                (log_normalizer, neg_trace, neg_log_det), z_est = model.neg_log_likelihood(x, u)
                loss = log_normalizer + neg_trace + neg_log_det

            loss.backward()
            optimizer.step()

            logger.update('loss', loss.item())
            if args.i_what == 'iFlow':
                logger.update('log_normalizer', log_normalizer.item())
                logger.update('neg_trace', neg_trace.item())
                logger.update('neg_log_det', neg_log_det.item())

            perf = mcc(z.cpu().numpy(), z_est.cpu().detach().numpy())
            logger.update('perf', perf)

            if acc_itr % args.log_freq == 0:  # % 25
                logger.log()
                writer.add_scalar('data/performance', logger.get_last('perf'), acc_itr)
                writer.add_scalar('data/loss', logger.get_last('loss'), acc_itr)

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
                           logger.get_last('perf'))

        epoch += 1
        eet = time.time()
        if args.i_what == 'iVAE':
            print('epoch {}: {:.4f}s;\tloss: {:.4f};\tperf: {:.4f}'.format(epoch,
                                                                           eet - est,
                                                                           logger.get_last('loss'),
                                                                           logger.get_last('perf')))
        elif args.i_what == 'iFlow':
            print('epoch {}: {:.4f}s;\tloss: {:.4f} (l1: {:.4f}, l2: {:.4f}, l3: {:.4f});\tperf: {:.4f}'.format( \
                epoch,
                eet - est,
                logger.get_last('loss'),
                logger.get_last('log_normalizer'),
                logger.get_last('neg_trace'),
                logger.get_last('neg_log_det'),
                logger.get_last('perf')))

    et = time.time()
    print('training time: {}s'.format(et - ste))

    # Save final model
    checkpoint(PT_MODELS_FOLDER,
               "",
               'final',
               model,
               optimizer,
               logger.get_last('loss'),
               logger.get_last('perf'))

    writer.close()
    if not args.no_log:
        logger.add_metadata(**metadata)
        logger.save_to_json()
        logger.save_to_npz()

    print('total time: {}s'.format(et - st))
    return model

def test_model(model, device, save_mcc=False):
    ###### Run Test Here
    model.eval()

    # Grab data arguments from dataset filename
    model_name = model.__class__.__name__
    if model_name == 'iFlow':
        data_args = model.args['file'].split('/')[-1][4:-4] + '_f'
        seed = model.args['seed']
        epochs = model.args['epochs']
    elif model_name == 'iVAE':
        data_args = model.file.split('/')[-1][4:-4] + '_f'
        seed = model.seed
        epochs = model.epochs


    data_file = create_if_not_exist_dataset(root='data/{}/'.format(seed), arg_str=data_args)
    A = np.load(data_file)

    x = A['x']  # of shape
    x = torch.from_numpy(x).to(device)
    print("x.shape ==", x.shape)

    s = A['s']  # of shape
    # s = torch.from_numpy(s).to(device)
    print("s.shape ==", s.shape)

    u = A['u']  # of shape
    u = torch.from_numpy(u).to(device)
    print("u.shape ==", u.shape)

    if model_name == 'iVAE':
        _, z_est = model.elbo(x, u)
    elif model_name == 'iFlow':
        # (_, _, _), z_est = model.neg_log_likelihood(x, u)
        total_num_examples = reduce(operator.mul, map(int, data_args.split('_')[:2]))
        model.set_mask(total_num_examples)
        z_est, nat_params = model.inference(x, u)

    z_est = z_est.cpu().detach().numpy()


    Z_EST_FOLDER = osp.join('z_est/', data_args + '_' + str(epochs))

    if not osp.exists(Z_EST_FOLDER):
        os.makedirs(Z_EST_FOLDER)
    np.save("{}/z_est_{}.npy".format(Z_EST_FOLDER, model_name), z_est)
    if model_name == 'iFlow':
        nat_params = nat_params.cpu().detach().numpy()
        np.save("{}/nat_params.npy".format(Z_EST_FOLDER), nat_params)
    print("z_est.shape ==", z_est.shape)

    perf = mcc(s, z_est)
    corr_coefs = correlation_coefficients(s, z_est)
    print("EVAL PERFORMANCE: {}".format(perf))

    if save_mcc:
        # Writes results for current model, flow type and n layers in mixing MLP
        # Saved as i-what_nsource_nlayers_flowlength_prior.txt
        if model_name == 'iVAE':
            with open(osp.join('results', "_".join([model_name, data_args]) + '.txt'), 'a+') as f:
                # with open(osp.join('results', "_".join([args.i_what, "_".join(args.data_args.split("_")[:2]),
                #                                         args.data_args.split("_")[4], args.data_args.split("_")[6]]) + '.txt'), 'a+') as f:
                f.write(", " + str(perf))
        elif model_name == 'iFlow':
            with open(osp.join('results', "_".join([model_name, data_args]) + '.txt'), 'a+') as f:
                # with open(osp.join('results', "_".join([args.i_what, "_".join(args.data_args.split("_")[:2]),
                #                                         args.data_args.split("_")[4],
                #                                         args.data_args.split("_")[6]]) + '.txt'), 'a+') as f:
                f.write(", " + str(perf))

    print("DONE.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default=None, help='path to data file in .npz format. (default None)')
    parser.add_argument('-x', '--data-args', type=str, default='1000_40_2_2_3_48_gauss_xtanh_u_f',
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
    parser.add_argument('-npa', '--nat_param_act', type=str, default="Softplus")
    parser.add_argument('-u', '--gpu_id', type=str, default='0')
    parser.add_argument('-fl', '--flow_length', type=int, default=10)
    parser.add_argument('-lr_df', '--lr_drop_factor', type=float, default=0.25)
    parser.add_argument('-lr_pn', '--lr_patience', type=int, default=10)
    parser.add_argument('-mcc', '--mcc_store', action='store_true', default=False, help='save MCC to path /results/i-what_nps_ns_nl')

    args = parser.parse_args()

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    EXPERIMENT_FOLDER = osp.join('experiments/', now)
    LOG_FOLDER = osp.join(EXPERIMENT_FOLDER, 'log/')
    TENSORBOARD_RUN_FOLDER = osp.join(EXPERIMENT_FOLDER, 'runs/')
    TORCH_CHECKPOINT_FOLDER = osp.join(EXPERIMENT_FOLDER, 'ckpt/')
    PT_MODELS_FOLDER = osp.join('pt_models/', args.data_args + '_' + args.i_what + '_' + str(args.epochs))
    Z_EST_FOLDER = osp.join('z_est/', args.data_args + '_' + str(args.epochs))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    print(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    st = time.time()

    if args.file is None:
        args.file = create_if_not_exist_dataset(root='data/{}/'.format(args.seed), arg_str=args.data_args)

    device = torch.device('cuda' if args.cuda else 'cpu')

    metadata = vars(args).copy()
    metadata.update({"device": device})


    model = train_model(args, metadata, device=device)
    test_model(model, device=device, save_mcc=args.save_mcc)