# Bohrium-AI4SCup-USCT

[AI4S Cup - 超声CT成像中的声场预测](https://bohrium.dp.tech/competitions/2512153120?tab=introduce) 正向解算私榜第二名方案。

## 特性

* 基于PyTorch的Dataset和Dataloader
* 基于`Hydra`的超参数管理
* 基于`PyTorch-Lightning`的多GPU训练
* 改进BornFNO

## 快速上手

### 环境搭建

#### 硬件环境

- RAM: > 32GB
- GPU: 6 x NVIDIA GTX 4090

#### 软件环境

- Ubuntu 22.04.3 LTS
- python 3.10
- pytorch 2.2.0: pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
- Other packages: `pip install -r requirements.txt`

### 最优解的日志

我已经将最优解的日志上传到`./best_log`文件夹中。

您可以通过`tensorboard`查看最优解的日志：
```bash
cd ./best_log
tensorboard --logdir=./
```

### 代码测试

您可以运行以下命令测试代码是否在您的机器上正常运行：

```bash
# ======
# 预处理数据
# 仅使用前10个速度场文件
# ======
cd ./scripts
python restore_data.py --data_folder /data/Comp/Bohrium-AI4SCup-CT-2024/data/helmholtz --extend_name v1_10 --n_max 10
# ------------------------------------------------
# 请手动将`u_homo.npy`文件移动到`./data`
# ------------------------------------------------

# =====
# 训练流程
# 仅使用一张GPU (GPU ID = 0)训练1个epoch
# =====
cd ../
python training.py data_conf.v_speed=v1_10 data_conf.v_field=v1_10 data_conf.v_dataset=v2 loss_conf.loss_type=rrmse data_conf.batch_size=5 train_conf.only_first_fold=True train_conf.epochs=1 model_type=bfnov3 fno_conf.lifting_size=60 fno_conf.wavenumber=[100,100,100,100,100,100,100] data_conf.preprocess=fixed optim_conf.lr=6e-3 optim_conf.weight_decay=0 fno_conf.activation=gelu fno_conf.use_bn=True fno_conf.simplified_fourier=True

# =====
# 推理
# =====
# ------------------------------------------------
# 您能够在`./outputs/XXXX-XX-XX/XX-XX-XX`文件夹中找到输出的模型文件`fold_0.ckpt`，该文件夹基于训练运行时间戳自动创建
# 将该模型文件拷贝到'./outputs/models'
# ------------------------------------------------
python inference.py
```

### 完整训练

#### 准备数据

```bash
cd ./scripts
python restore_data.py --data_folder /data4/yzj/data/AI4S/helmholtz --extend_name v1_7200
```

请将`--data_folder`替换为您自己本地的源数据文件夹，预处理后的数据将保存在`./data`文件夹中。

此外，您需要手动将`u_homo.npy`移动到`./data`文件夹。


#### 训练

您可以使用以下命令复现我的方案：

```bash
python training.py device=2+3+4+5+6+7 data_conf.v_speed=v1_7200 data_conf.v_field=v1_7200 data_conf.v_dataset=v2 loss_conf.loss_type=rrmse data_conf.batch_size=5 train_conf.only_first_fold=True train_conf.epochs=30 model_type=bfnov3 fno_conf.lifting_size=60 fno_conf.wavenumber=[100,100,100,100,100,100,100] data_conf.preprocess=fixed optim_conf.lr=6e-3 optim_conf.weight_decay=0 fno_conf.activation=gelu fno_conf.use_bn=True fno_conf.simplified_fourier=True
```

请将`device`替换为您的设备id列表，例如`device=0+1+2+3+4`表示使用5张卡训练，其ID分别为0,1,2,3,4。

本解决方案大概需要在6张4090上训练两天。日志文件以及模型文件将保存到`./outputs`文件夹。


#### 推理

为了进行推理测试以及生成提交文件`submission.pt`。您需要将`fold_0.ckpt`文件移动到`./outputs/models`并运行以下命令：
```bash
python inference.py
```


## 致谢

本项目基于比赛官方的基线代码。感谢主办方举办本次比赛。此外，也感谢课题组的算力支持。
