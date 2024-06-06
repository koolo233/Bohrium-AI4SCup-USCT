import glob
import os
import argparse
import numpy as np
from tqdm import tqdm


parse = argparse.ArgumentParser()
parse.add_argument('--data_folder', type=str, default='../data', help='Path to the speed data')
parse.add_argument('--output_folder', type=str, default='../data', help='Path to save the data')
parse.add_argument('--extend_name', type=str, default='v1', help='Name of the output file')
parse.add_argument('--n_max', type=int, default=7200, help='Maximum number of data to be loaded')
parse.add_argument('--debug', action='store_true', help='Debug mode')
parse.add_argument('--display', action='store_true', help='Display the data')
args = parse.parse_args()


if __name__ == '__main__':

    speed_array = np.zeros((args.n_max, 480, 480), dtype=np.float32)
    field_array = np.zeros((args.n_max * 32, 480, 480, 2), dtype=np.float32)

    for sample_id in tqdm(range(1, args.n_max+1)):

        sub_folder_id = (sample_id-1) // 900 + 1

        data_folder = os.path.join(args.data_folder, f'dataset_train_{sub_folder_id}')
        speed_data_path = os.path.join(data_folder, 'speed', f'train_{sample_id}.npy')
        field_data_paths = [os.path.join(data_folder, 'field', f'train_{sample_id}_{i}.npy') for i in range(1, 5)]

        speed_data = np.load(speed_data_path).astype(np.float32)
        field_data = np.concatenate([np.load(field_data_path) for field_data_path in field_data_paths], axis=0)
        bs, n_x, n_y = field_data.shape
        data_real = np.real(field_data).astype(np.float32).reshape((bs, n_x, n_y, 1))
        data_imag = np.imag(field_data).astype(np.float32).reshape((bs, n_x, n_y, 1))

        speed_array[sample_id-1] = speed_data
        field_array[(sample_id-1)*32:sample_id*32] = np.concatenate([data_real, data_imag], axis=-1)

        if args.debug:
            break

    if args.display:
        import matplotlib.pyplot as plt
        plot_key_list = range(1 if args.debug else 4)
        print(plot_key_list)
        fig = plt.figure(figsize=(10, 5))
        for key in plot_key_list:

            # speed
            ax = fig.add_subplot(3, len(plot_key_list), plot_key_list.index(key) + 1, xticks=[], yticks=[])
            ax.imshow(speed_array[key], cmap='jet')
            ax.set_title(f'ID: {key}, Speed')

            # field
            ax = fig.add_subplot(3, len(plot_key_list), plot_key_list.index(key) + 1 + len(plot_key_list), xticks=[], yticks=[])
            ax.imshow(field_array[key * 32, :, :, 0], cmap='RdBu_r')
            ax.set_title(f'ID: {key}, Real')

            ax = fig.add_subplot(3, len(plot_key_list), plot_key_list.index(key) + 1 + 2*len(plot_key_list), xticks=[], yticks=[])
            ax.imshow(field_array[key * 32, :, :, 1], cmap='RdBu_r')
            ax.set_title(f'ID: {key}, Imag')

        plt.tight_layout()
        plt.show()

    # save to npy
    os.makedirs(args.output_folder, exist_ok=True)
    speed_save_path = os.path.join(args.output_folder, f'speed_only_data_{args.extend_name}.npy')
    field_save_path = os.path.join(args.output_folder, f'field_only_data_{args.extend_name}.npy')
    np.save(speed_save_path, speed_array)
    np.save(field_save_path, field_array)
    print('save speed data to ', speed_save_path)
    print('save field data to ', field_save_path)
