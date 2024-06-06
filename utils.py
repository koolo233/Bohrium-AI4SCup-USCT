import os
import glob
import torch
import numpy as np
from hydra.utils import get_original_cwd

from dataset import CustomDatasetV1, CustomDatasetV2
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


def show_batch(ds):
    fig = plt.figure(figsize=(24, 24))
    sample_idx = np.random.randint(0, len(ds) - 1, 1)

    input_tensor, theta_tensor, target_tensor = ds[sample_idx[0]]
    input_tensor = input_tensor.detach().numpy()  # [480, 480, 1]
    theta_tensor = theta_tensor.detach().numpy()  # [480, 480, 3]
    target_tensor = target_tensor.detach().numpy()  # [480, 480, 2]

    ax = fig.add_subplot(2, 3, 1, xticks=[], yticks=[])
    ax.imshow(input_tensor[:, :, 0], cmap='jet')
    ax.set_title('Input Data')

    for i in range(theta_tensor.shape[-1]):
        ax = fig.add_subplot(2, 3, i + 2, xticks=[], yticks=[])
        ax.imshow(theta_tensor[:, :, i], cmap='RdBu_r')
        ax.set_title(f'Theta {i}')

    for i in range(2):
        ax = fig.add_subplot(2, 3, i + 5, xticks=[], yticks=[])
        ax.imshow(target_tensor[:, :, i], cmap='RdBu_r')
        ax.set_title(f'Target {i}')

    # plt.show()
    plt.tight_layout()
    plt.savefig('batch_data.png')


def load_data(cfg):

    data_cfg = cfg['data_conf']
    data_root = data_cfg['data_root']

    speed_data_file = f"{get_original_cwd()}/{data_root}/speed_{data_cfg['v_speed']}.npy"
    field_data_files = glob.glob(f"{get_original_cwd()}/{data_root}/field_{data_cfg['v_field']}_*.npy")
    src_data_file = f'{get_original_cwd()}/{data_root}/u_homo.npy'

    speed_data = np.load(speed_data_file, allow_pickle=True).item()
    src_data = np.load(src_data_file)
    field_data = dict()
    for field_data_file in field_data_files:
        print(f'load {field_data_file}')
        field_sub_id = int(os.path.basename(field_data_file).split('.')[0].split('_')[-1]) - 1
        field_data_tmp = np.load(field_data_file, allow_pickle=True).item()
        for key, val in field_data_tmp.items():
            for sub_id in range(8):
                new_key = f'{key}-{sub_id + field_sub_id * 8}'
                field_data[new_key] = val[sub_id, ...]

    return speed_data, src_data, field_data


def load_data_v2(cfg):
    data_cfg = cfg['data_conf']
    data_root = data_cfg['data_root']

    speed_data_file = f"{get_original_cwd()}/{data_root}/speed_only_data_{data_cfg['v_speed']}.npy"
    field_data_file = f"{get_original_cwd()}/{data_root}/field_only_data_{data_cfg['v_field']}.npy"
    src_data_file = f'{get_original_cwd()}/{data_root}/u_homo.npy'

    speed_data = np.load(speed_data_file, mmap_mode='r')
    src_data = np.load(src_data_file)
    field_data = np.load(field_data_file, mmap_mode='r')

    return speed_data, src_data, field_data


def get_fold_dls(cfg, df_train, df_valid, all_speed_data, all_src_data, all_field_data):

    data_cfg = cfg['data_conf']

    if data_cfg['v_dataset'] == 'v1':
        dataset_cls = CustomDatasetV1
    elif data_cfg['v_dataset'] == 'v2':
        dataset_cls = CustomDatasetV2
    else:
        raise

    ds_train = dataset_cls(
        cfg=data_cfg,
        metadata=df_train,
        input_data_array=all_speed_data,
        target_data_array=all_field_data,
        src_data_array=all_src_data,
        mode='train'
    )

    ds_val = dataset_cls(
        cfg=data_cfg,
        metadata=df_valid,
        input_data_array=all_speed_data,
        target_data_array=all_field_data,
        src_data_array=all_src_data,
        mode='valid'
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=data_cfg['batch_size'],
        shuffle=True,
        num_workers=data_cfg['num_workers'],
        pin_memory=data_cfg['pin_memory'],
        persistent_workers=data_cfg['persistent_workers']
    )

    dl_val = DataLoader(
        ds_val,
        batch_size=data_cfg['batch_size'],
        shuffle=False,
        num_workers=data_cfg['num_workers'],
        pin_memory=data_cfg['pin_memory'],
        persistent_workers=data_cfg['persistent_workers']
    )

    return dl_train, dl_val, ds_train, ds_val


class RMSE:

    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, x, y):
        bs = x.shape[0]

        diff_norms = torch.norm(x.reshape(bs, -1) - y.reshape(bs, -1), 2, dim=-1)

        if self.reduction == 'mean':
            return torch.mean(diff_norms)
        elif self.reduction == 'sum':
            return torch.sum(diff_norms)
        else:
            raise Exception(f'Reduction {self.reduction} not recognized')


class RelateRMSE:

    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, x, y):
        bs = x.shape[0]

        diff_norms = torch.norm(x.reshape(bs, -1) - y.reshape(bs, -1), 2, dim=-1)
        y_norms = torch.norm(y.reshape(bs, -1), 2, dim=-1)

        if self.reduction == 'mean':
            return torch.mean(diff_norms / y_norms)
        elif self.reduction == 'sum':
            return torch.sum(diff_norms / y_norms)
        else:
            raise Exception(f'Reduction {self.reduction} not recognized')
