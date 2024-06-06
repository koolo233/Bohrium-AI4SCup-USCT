import glob
import os.path

import matplotlib.pyplot as plt
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np


class CustomDatasetV1(Dataset):

    def __init__(
            self,
            cfg,
            metadata,
            input_data_array,
            target_data_array,
            src_data_array,
            mode='train'
    ):
        super().__init__()

        self.cfg = cfg
        self.mode = mode
        self.metadata = metadata
        self.input_data_array = input_data_array
        self.target_data_array = target_data_array
        self.src_data_array = src_data_array

        self.target_data_array_keys = list(self.target_data_array.keys())

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        meta_row = self.metadata.iloc[item]
        data_key = meta_row['sample_key']  # data_id - sequence_id

        sample_id, seq_id = data_key.split('-')

        # input data
        input_data = self.input_data_array[sample_id]  # [h, w]
        if self.cfg['preprocess'] == 'fixed':
            input_data = (1500 / input_data - 1) * 30
        h, w = input_data.shape
        input_data = torch.tensor(input_data, dtype=torch.float32).view(h, w, 1)

        # target data
        target_data = self.target_data_array[data_key]  # [h, w, 2]
        if self.cfg['preprocess'] == 'fixed' and self.mode == 'train':
            target_data = target_data * 2e-3
        target_data = torch.tensor(target_data, dtype=torch.float32).view(h, w, 2)

        # source info
        # theta = (int(seq_id)/32 * 2 * np.pi) * np.ones((h, w, 1))  # [h, w, 1]
        src = self.src_data_array[int(seq_id)].reshape(h, w, 2)  # [h, w, 2]
        src[:, :, 1] = - src[:, :, 1]  # flip the y-axis
        if self.cfg['preprocess'] == 'fixed':
            src = src * 2e-3
        # theta = np.concatenate([theta, src], axis=-1)

        return input_data, torch.tensor(src, dtype=torch.float32), target_data


class CustomDatasetV2(Dataset):

    def __init__(
            self,
            cfg,
            metadata,
            input_data_array,
            target_data_array,
            src_data_array,
            mode='train'
    ):
        super().__init__()

        self.cfg = cfg
        self.mode = mode
        self.metadata = metadata
        self.input_data_memmap_array = input_data_array
        self.target_data_memmap_array = target_data_array
        self.src_data_array = src_data_array

    def __len__(self):
        return len(self.metadata) * 32

    def __getitem__(self, item):

        sample_id = item // 32
        sub_sample_id = item % 32

        # input data
        input_data = np.copy(self.input_data_memmap_array[sample_id])  # [h, w]
        if self.cfg['preprocess'] == 'fixed':
            input_data = (1500 / input_data - 1) * 30
        h, w = input_data.shape
        input_data = torch.tensor(input_data, dtype=torch.float32).view(h, w, 1)

        # target data
        target_data = np.copy(self.target_data_memmap_array[item])  # [h, w, 2]
        if self.cfg['preprocess'] == 'fixed' and self.mode == 'train':
            target_data = target_data * 2e-3
        target_data = torch.tensor(target_data, dtype=torch.float32).view(h, w, 2)

        # source info
        theta = (int(sub_sample_id)/32 * 2 * np.pi) * np.ones((h, w, 1))  # [h, w, 1]
        src = self.src_data_array[sub_sample_id].reshape(h, w, 2)  # [h, w, 2]
        src[:, :, 1] = - src[:, :, 1]  # flip the y-axis
        if self.cfg['preprocess'] == 'fixed':
            src = src * 2e-3
            theta = theta / (2 * np.pi)
        theta = np.concatenate([theta, src], axis=-1)

        return input_data, torch.tensor(theta, dtype=torch.float32), target_data


class CustomIterableDatasetV1(IterableDataset):

    def __init__(
            self,
            cfg,
            metadata
    ):
        super().__init__()

        self.cfg = cfg
        self.metadata = metadata

        self.buffer = list()
        self.buffer_size = self.cfg['buffer_size']
        self.shuffle = self.cfg['shuffle']

    def __iter__(self):
        # obtain the worker info
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        # split metadata
        metadata_len = len(self.metadata)
        split_len = metadata_len // num_workers
        start_idx = worker_id * split_len
        end_idx = metadata_len if worker_id == num_workers - 1 else (worker_id + 1) * split_len
        current_metadata = self.metadata[start_idx:end_idx]

        # load data to buffer
        for meta_info in current_metadata:
            for sample in self.load_samples(meta_info):
                if len(self.buffer) >= self.buffer_size: # buffer is full
                    yield self.buffer.pop(0)
                self.buffer = np.concatenate([self.buffer, [sample]], axis=0)

    def load_samples(self, meta_info):
        # load samples
        # input: speed
        speed_data = np.load(meta_info['speed_path'])  # [h, w]
        # target: field
        for field_path in meta_info['field_paths']:
            field_data = np.load(field_path)  # [B, h, w]
            b, h, w = field_data.shape
            filed_real = np.real(field_data).astype(np.float32).reshape([b, h, w, 1])  # [B, h, w]
            filed_imag = np.imag(field_data).astype(np.float32).reshape([b, h, w, 1])  # [B, h, w]
            field_data = np.concatenate([filed_real, filed_imag], axis=-1)
            yield speed_data, field_data


if __name__ == '__main__':
    speed_data_file = './data/speed_v1.npy'
    field_data_files = glob.glob('./data/field_v1_*.npy')
    src_data_file = './data/u_homo.npy'

    speed_data = np.load(speed_data_file, allow_pickle=True).item()
    src_data = np.load(src_data_file)
    field_data = dict()
    for field_data_file in field_data_files:
        print(f'load {field_data_file}')
        field_sub_id = int(os.path.basename(field_data_file).split('.')[0].split('_')[-1]) - 1
        field_data_tmp = np.load(field_data_file, allow_pickle=True).item()
        for key, val in field_data_tmp.items():
            for sub_id in range(8):
                new_key = f'{key}-{sub_id+field_sub_id*8}'
                field_data[new_key] = val[sub_id, ...]

    test_dataset = CustomDatasetV1({}, input_data_array=speed_data, target_data_array=field_data, src_data_array=src_data)

    fig = plt.figure(figsize=(24, 24))
    sample_idx = np.random.randint(0, len(test_dataset)-1, 1)

    input_tensor, theta_tensor, target_tensor = test_dataset[sample_idx[0]]
    input_tensor = input_tensor.detach().numpy()  # [480, 480, 1]
    theta_tensor = theta_tensor.detach().numpy()  # [480, 480, 3]
    target_tensor = target_tensor.detach().numpy()  # [480, 480, 2]

    ax = fig.add_subplot(2, 3, 1, xticks=[], yticks=[])
    ax.imshow(input_tensor[:, :, 0], cmap='jet')
    ax.set_title('Input Data')

    for i in range(3):
        ax = fig.add_subplot(2, 3, i+2, xticks=[], yticks=[])
        ax.imshow(theta_tensor[:, :, i], cmap='RdBu_r')
        ax.set_title(f'Theta {i}')

    for i in range(2):
        ax = fig.add_subplot(2, 3, i+5, xticks=[], yticks=[])
        ax.imshow(target_tensor[:, :, i], cmap='RdBu_r')
        ax.set_title(f'Target {i}')

    plt.show()
