import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import torch
import os
from lib.data import create_if_not_exist_dataset
from lib.iFlow import iFlow
from lib.models import iVAE
from lib.metrics import mean_corr_coef as mcc

from scipy.optimize import linear_sum_assignment


def correlation_coefficients(x, y, method='pearson'):
    """
    A numpy implementation of the mean correlation coefficient metric.

    :param x: numpy.ndarray
    :param y: numpy.ndarray
    :param method: str, optional
            The method used to compute the correlation coefficients.
    :return: float
    """
    d = x.shape[1]
    if method == 'pearson':
        cc = np.corrcoef(x, y, rowvar=False)[:d, d:]
    else:
        raise ValueError('not a valid method: {}'.format(method))
    cc_matrix = np.abs(cc)

    corr_coefs = cc_matrix[linear_sum_assignment(-1 * cc_matrix)] #.mean()
    return corr_coefs

def plot_2d(s, x, u, z_est_iFlow, z_est_iVAE, iFlow_perf=None, iVAE_perf=None, filename=None):
    """
    s : true latent variables, source of the observations
    x : observations, the inputs of iFlow and iVAE models
    z_est_iFlow : predictions of the iFlow model
    z_est_iVAE : predictions of the iVAE model
    iFlow_perf : .txt file containing iFlow scores
    iVAE_perf : .txt file containing iVAE scores
    filename : targed name for saving the file in results/2D_visualizations
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    fig.tight_layout()

    n, n_segments = u.shape
    n_per_segment = int(n / n_segments)
    for i in range(n_segments):
        (start, stop) = i * n_per_segment, (i + 1) * n_per_segment
        ax1.scatter(s[start:stop, 0], s[start:stop, 1], s=1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xlabel("Original sources")
        ax2.scatter(x[start:stop, 0], x[start:stop, 1], s=1)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlabel("Observations")
        ax3.scatter(z_est_iFlow[start:stop, 0], z_est_iFlow[start:stop, 1], s=1)
        ax3.set_xticks([])
        ax3.set_yticks([])
        if iFlow_perf:
            ax3.set_xlabel("iFlow (MCC: {:.2f})".format(iFlow_perf))
        else:
            ax3.set_xlabel("iFlow")
        ax4.scatter(z_est_iVAE[start:stop, 0], z_est_iVAE[start:stop, 1], s=1)
        ax4.set_xticks([])
        ax4.set_yticks([])
        if iVAE_perf:
            ax4.set_xlabel("iVAE (MCC: {:.2f})".format(iVAE_perf))
        else:
            ax4.set_xlabel("iVAE")
    plt.show()
    fig.savefig('results/2D_visualizations/' + filename)
    return


def load_plot_2d(seeds, data_arguments='1000_5_2_2_$mixing-layers_$seed_gauss_xtanh_u_f', iFlow_results_file=None,
                 iVAE_results_file=None, epochs=20, mixing_layers=3):
    """
    seeds : list of dataset seeds for visualization
    data_arguments : arguments for the dataset, 'nps_ns_dl_dd_nl_s_p_a_u_n'
    iFlow_results_file : filename of corresponding iFlow results
    iVAE_results_file : filename of corresponding iVAE results
    epochs : number of training epochs
    mixing_layers : number of mixing layers used that generated the results
    """
    iFlow_perfs = None
    iVAE_perfs = None
    data_arguments = data_arguments.split("_")
    data_arguments = data_arguments[:4] + [str(mixing_layers)] + data_arguments[5:]

    print("Number of layers in dataset mixing MLP: ", mixing_layers)
    if iFlow_results_file:
        with open(iFlow_results_file) as f:
            iFlow_perfs = list(map(eval, f.readline().split(',')[1:]))
            print('iFlow mean = {:.4f}, std = {:.4f}'.format(np.mean(iFlow_perfs), np.std(iFlow_perfs)))
            print('len iFlow array:', len(iFlow_perfs))

    if iVAE_results_file:
        with open(iVAE_results_file) as f:
            iVAE_perfs = list(map(eval, f.readline().split(',')[1:]))
            print('iVAE mean = {:.4f}, std = {:.4f}'.format(np.mean(iVAE_perfs), np.std(iVAE_perfs)))
            print('len iVAE array:', len(iVAE_perfs))



    for i, seed in enumerate(seeds):
        data_arguments[5] = str(seed)
        data_file = create_if_not_exist_dataset(root='data/{}/'.format(1), arg_str="_".join(data_arguments))
        # load data
        path_to_dataset = data_file
        print('Dataset seed = {}'.format(seed))
        with np.load(path_to_dataset) as data:
            x = data['x']
            u = data['u']
            s = data['s']
        # load predictions
        path_to_z_est = "z_est/" + "_".join(data_arguments[:5]) + "_" + str(seed) + "_" + "_".join(
            data_arguments[6:]) + '_' + str(epochs) + "/"
        z_est_iFlow = np.load(path_to_z_est + "z_est_iFlow.npy")
        z_est_iVAE = np.load(path_to_z_est + "z_est_iVAE.npy")

        # Plotted figure is saved with data_args as filename in results/2D_visualizations/
        fig_name = "_".join(["_".join(data_arguments[:5]), str(seed), "_".join(data_arguments[6:]), str(epochs)])
        # plot and save figure
        plot_2d(s, x, u, z_est_iFlow, z_est_iVAE, iFlow_perfs[i], iVAE_perfs[i], filename=fig_name)


def load_model_from_checkpoint(ckpt_path, device, model_seed=1):
    print('checkpoint path:', ckpt_path)
    model_args = ckpt_path.split('/')[1]  # get folder name containing model and data properties
    ckpt_filename = ckpt_path.split('/')[-1]

    model_args = model_args.split('_')
    epochs = model_args[-1]
    model_name = model_args[-2]
    data_args = model_args[:-2]

    data_file = create_if_not_exist_dataset(root='data/{}/'.format(model_seed), arg_str="_".join(data_args))

    nps = int(data_args[0])
    ns = int(data_args[1])
    aux_dim = int(data_args[1])
    n = nps * ns
    latent_dim = int(data_args[2])
    data_dim = int(data_args[3])

    print('Loading model', model_name)
    model_path = ckpt_path

    print('Loading data', data_file)
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

    checkpoint = torch.load(model_path)

    # Arguments (metadata, from argparse in main.py), have to correspond to selected dataset and model properties
    # Hyperparameter and configurations as precribed in the paper.
    metadata = {'file': data_file, 'path': data_file, 'batch_size': 64,
                'epochs': epochs, 'device': device, 'seed': 1, 'i_what': model_name,

                'max_iter': None, 'hidden_dim': 50, 'depth': 3,
                'lr': 1e-3, 'cuda': True, 'preload': True,
                'anneal': False, 'log_freq': 25,
                'flow_type': 'RQNSF_AG', 'num_bins': 8,
                'nat_param_act': 'Softplus', 'gpu_id': '0',
                'flow_length': 10, 'lr_drop_factor': 0.25,
                'lr_patience': 10}

    # Get dataset properties
    metadata.update({'nps': nps, 'ns': ns, 'n': n, 'latent_dim': latent_dim, 'data_dim': data_dim, 'aux_dim': aux_dim})

    if model_name == 'iFlow':
        model = iFlow(args=metadata).to(device)
    elif model_name == "iVAE":
        model = iVAE(latent_dim,  # latent_dim
                     data_dim,  # data_dim
                     aux_dim,  # aux_dim
                     n_layers=metadata['depth'],
                     activation='lrelu',
                     device=device,
                     hidden_dim=metadata['hidden_dim'],
                     anneal=metadata['anneal'],  # False
                     file=metadata['file'],
                     seed=1)

    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def plot_5d_correlations(z_est_dataset_dir, show_iFlow=True, show_iVAE=True):
    data_arguments = z_est_dataset_dir.split('/')[-1]
    data_arguments = "_".join(data_arguments.split("_")[:-1])  # Remove last "_epochs" from string
    seed = data_arguments.split("_")[5]

    # load data
    path_to_dataset = "data/1/tcl_" + data_arguments[:-2] + ".npz"  # slice off "_f"
    with np.load(path_to_dataset) as data:
        x = data['x']
        u = data['u']
        s = data['s']
    # load predictions
    z_est_iFlow = None
    z_est_iVAE = None
    if show_iFlow:
        z_est_iFlow = np.load(osp.join(z_est_dataset_dir, "z_est_iFlow.npy").replace("\\", "/"))
    if show_iVAE:
        z_est_iVAE = np.load(osp.join(z_est_dataset_dir, "z_est_iVAE.npy").replace("\\", "/"))

    model_names = ['iFlow', 'iVAE']
    graph_color = ['y', 'indianred']

    n, n_segments = u.shape
    points_per_seg = n // n_segments

    for model_idx, z_est in enumerate([z_est_iFlow, z_est_iVAE]):
        if z_est is not None:
            print('{} performance on dataset'.format(model_names[model_idx]))
            print('Dataset seed = ', seed)

            # Get correlation coefficients
            print('MCC = ', round(mcc(s, z_est), 4))
            corr_coefs = correlation_coefficients(s, z_est)

            mean_source = np.zeros((5, n_segments))
            mean_est = np.zeros((5, n_segments))

            for i in range(n_segments):
                for dim in range(5):
                    mean_source[dim][i] = np.mean(s[i * points_per_seg:(i + 1) * points_per_seg, [dim]])
                    mean_est[dim][i] = np.mean(z_est[i * points_per_seg:(i + 1) * points_per_seg, [dim]])

            # Standardize both signals (mean=0, std=1)
            mean_source = (mean_source - np.mean(mean_source, axis=0)) / np.std(mean_source, axis=0)
            mean_est = (mean_est - np.mean(mean_est, axis=0)) / np.std(mean_est, axis=0)

            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            fig.tight_layout()
            plt.gcf().subplots_adjust(bottom=0.15, left=0.05)

            for i in range(5):
                axes[i].plot(mean_source[i], linestyle='-.')
                axes[i].plot(mean_est[i], linestyle='-.', color=graph_color[model_idx])
                axes[i].set_title("corr: " + str(round(corr_coefs[i], 4)))

            axes[2].set_xlabel('Segment')
            axes[0].set_ylabel('Latent Value')
            plt.show()

            fig.savefig('results/mcc_across_dims/' + "_".join([model_names[model_idx], data_arguments]))

def print_cc_mean_std(file):
    """
    file : path to data file in .txt format. (default None)
    """
    with open(file, 'r') as f:
        coeffs = f.read().replace('\n', '').split(';')[1:]

    array = np.array([c.strip('[]').split() for c in coeffs]).astype('float')
    std = array.std(axis=0)
    mean = array.mean(axis=0)
    print('means: ', np.round(mean, 4))
    print('standard deviations: ', np.round(std, 4))

def read_energy_values_from_tensorboard():
    energy_values = []
    for directory in os.listdir('experiments'):
        npz = np.load(os.path.join('experiments', directory, 'log', 'data', '1.npz'))
        # 'log_normalizer', 'neg_log_det', 'neg_trace', 'loss', 'perf'
        energy_values.append(-npz['loss'][-1])

    np.save(os.path.join('results', 'energy_values.npy'), np.array(energy_values))