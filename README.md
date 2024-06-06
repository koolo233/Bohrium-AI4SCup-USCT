# Bohrium-AI4SCup-USCT
The public code of 2nd solution (forward problem, private) of [Bohrium AI4S Cup USCT](https://bohrium.dp.tech/competitions/2512153120?tab=introduce).

* README: [简体中文](./README_zn.md)

## Features

* PyTorch's Dataset & Dataloader
* Manage hyper-parameters via `Hydra`
* Implement Multi-GPU Training via `PyTorch-Lightning`
* Adaptive BornFNO

## Quick Start

### ENV

#### Hardware

- RAM: > 32GB
- GPU: 6 x NVIDIA GTX 4090

#### Software

- Ubuntu 22.04.3 LTS
- python 3.10
- pytorch 2.2.0: pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
- Other packages: `pip install -r requirements.txt`

### Log of Best Solution

I have uploaded the log of the best solution in the `./best_log` folder. 

You can view the log of the best solution via `tensorboard`:
```bash
cd ./best_log
tensorboard --logdir=./
```

### Code Testing

You can run the following command to test the code if it works well on your machine:

```bash
# ======
# prepare data
# only use the first 10 speed files
# ======
cd ./scripts
python restore_data.py --data_folder /data/Comp/Bohrium-AI4SCup-CT-2024/data/helmholtz --extend_name v1_10 --n_max 10
# ------------------------------------------------
# please manually move the `u_homo.npy` to `./data`
# ------------------------------------------------

# =====
# training
# only use 1 GPU (GPU ID = 0) and training for 1 epoch
# =====
cd ../
python training.py data_conf.v_speed=v1_10 data_conf.v_field=v1_10 data_conf.v_dataset=v2 loss_conf.loss_type=rrmse data_conf.batch_size=5 train_conf.only_first_fold=True train_conf.epochs=1 model_type=bfnov3 fno_conf.lifting_size=60 fno_conf.wavenumber=[100,100,100,100,100,100,100] data_conf.preprocess=fixed optim_conf.lr=6e-3 optim_conf.weight_decay=0 fno_conf.activation=gelu fno_conf.use_bn=True fno_conf.simplified_fourier=True

# =====
# inference
# =====
# ------------------------------------------------
# you can find the `fold_0.ckpt` file from `./outputs/XXXX-XX-XX/XX-XX-XX` folder, which is named with runtime timestamps
# copy this file to './outputs/models'
# ------------------------------------------------
python inference.py
```

### Fully Training

#### Prepare Data

```bash
cd ./scripts
python restore_data.py --data_folder /data4/yzj/data/AI4S/helmholtz --extend_name v1_7200
```

Please replace the `--data_folder` with your own data folder. The pre-processed data will be saved in the `./data` folder.

In addition, you should manually move the `u_homo.npy` to `./data` folder.


#### Training

To reproduce the solution, you can run the following command:

```bash
python training.py device=2+3+4+5+6+7 data_conf.v_speed=v1_7200 data_conf.v_field=v1_7200 data_conf.v_dataset=v2 loss_conf.loss_type=rrmse data_conf.batch_size=5 train_conf.only_first_fold=True train_conf.epochs=30 model_type=bfnov3 fno_conf.lifting_size=60 fno_conf.wavenumber=[100,100,100,100,100,100,100] data_conf.preprocess=fixed optim_conf.lr=6e-3 optim_conf.weight_decay=0 fno_conf.activation=gelu fno_conf.use_bn=True fno_conf.simplified_fourier=True
```

Please replace the `device` with your own GPU device ids, for example, `device=0+1+2+3+4` for training on 5 GPUs (GPU ID = 0,1,2,3,4).

The solution costs about 2 days to train on 6 NVIDIA GTX 4090. Logging files and ckpt will be saved in the `./outputs` folder.


#### Inference

To test inference and generate the `submission.pt`, you should move the `fold_0.ckpt` to `./outputs/models` and run the following command:

```bash
python inference.py
```


## Acknowledgement

This project is based on the Baseline code provided by Bohrium AI4S Cup USCT. I would like to thank the organizers for the opportunity to participate in this competition. In addition, I would like to thank the laboratory for providing the necessary hardware resources for training.
