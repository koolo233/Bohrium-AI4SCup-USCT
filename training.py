import os.path

import hydra
import logging
from omegaconf import OmegaConf
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import GroupKFold

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar


from model import BornFNOV3
from utils import *


class TrainingSystem(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # == model ==
        if cfg['model_type'] == 'fno':
            model_cfg = cfg['fno_conf']
            self.model = FNO(model_cfg)
        elif cfg['model_type'] == 'afno':
            model_conf = cfg['afno_conf']
            self.model = AFNO2D(**model_conf)
        elif cfg['model_type'] == 'hrnet':
            model_conf = cfg['hrnet_conf']
            self.model = MyHRNet(model_conf)
        elif cfg['model_type'] == 'bfno':
            model_conf = cfg['fno_conf']
            self.model = BornFNO(model_conf)
        elif cfg['model_type'] == 'bfnov2':
            model_conf = cfg['fno_conf']
            self.model = BornFNOV2(model_conf)
        elif cfg['model_type'] == 'bfnov3':
            model_conf = cfg['fno_conf']
            self.model = BornFNOV3(model_conf)
        elif cfg['model_type'] == 'kanbfno':
            model_conf = cfg['kanfno_conf']
            self.model = KANBornFNO(model_conf)
        elif cfg['model_type'] == 'nbso':
            model_conf = cfg['fno_conf']
            self.model = NBSO(model_conf)

        # == loss ==
        if self.cfg['loss_conf']['loss_type'] == 'l1':
            self.loss = nn.L1Loss()
        elif self.cfg['loss_conf']['loss_type'] == 'l2':
            self.loss = nn.MSELoss()
        elif self.cfg['loss_conf']['loss_type'] == 'smooth_l1':
            self.loss = nn.SmoothL1Loss()
        elif self.cfg['loss_conf']['loss_type'] == 'rrmse':
            self.loss = RelateRMSE()
        else:
            raise Exception('Loss function not recognized from the list')

        # == train ==
        self.train_cfg = cfg['train_conf']

        self.optimizer = None

        # == validation ==
        self.val_score_fn = RelateRMSE()
        self.validation_step_outputs = list()
        self.val_score = 1000

    def forward(self, input_data, src_data):
        return self.model(input_data, src_data)

    def training_step(self, batch, batch_idx):
        # = data =
        input_data, src_data, target_data = batch
        bs = input_data.shape[0]

        # = forward =
        pred = self(input_data, src_data)
        loss = self.loss(
            pred.view(bs, -1),
            target_data.view(bs, -1)
        )

        # = log =
        self.log('train_loss', loss, True, sync_dist=True)
        self.log('lr', self.optimizer.param_groups[0]['lr'], True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):

        self.model.eval()

        # = data =
        input_data, src_data, target_data = batch

        # = forward =
        with torch.no_grad():
            pred = self(input_data, src_data)

        if self.cfg['data_conf']['preprocess'] == 'fixed':
            loss = self.loss(pred / 2e-3, target_data)
        else:
            loss = self.loss(pred, target_data)

        self.validation_step_outputs.append({'val_loss': loss})

        # self.log('val_loss', loss, True, sync_dist=True)

        self.model.train()

        return loss

    def on_validation_epoch_end(self):

        # https://github.com/Lightning-AI/pytorch-lightning/discussions/13041

        all_validation_outputs = self.all_gather(self.validation_step_outputs)

        # if self.trainer.is_global_zero:
        val_score = torch.stack([
            item['val_loss'] for item in all_validation_outputs
        ]).mean()

        if val_score < self.val_score:
            self.val_score = val_score

        # self.trainer.strategy.barrier()

        self.validation_step_outputs = list()

        self.log('val_loss', val_score, True, sync_dist=True)

    def configure_optimizers(self):

        optim_conf = self.cfg['optim_conf']

        # get trained parameters
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        # optim
        if optim_conf['type'] == 'adam':
            optimizer = torch.optim.Adam(trainable_params, lr=optim_conf['lr'],
                                         weight_decay=optim_conf['weight_decay'])
        elif optim_conf['type'] == 'adamw':
            optimizer = torch.optim.AdamW(trainable_params, lr=optim_conf['lr'],
                                          weight_decay=optim_conf['weight_decay'])
        elif optim_conf['type'] == 'sgd':
            optimizer = torch.optim.SGD(trainable_params, lr=optim_conf['lr'],
                                        weight_decay=optim_conf['weight_decay'], nesterov=True, momentum=0.9)
        else:
            raise Exception('Optimizer not recognized from the list')

        # sch
        if optim_conf['scheduler'] == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.train_cfg['epochs'],
                T_mult=1,
                eta_min=1e-6,
                last_epoch=-1
            )
        else:
            raise Exception('Scheduler not recognized from the list')

        self.optimizer = optimizer

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val_loss',
            }
        }


def prediction(dl, model):
    model.eval()
    criterion = RMSE(reduction='sum')
    total_loss = 0
    n_samples = 0
    for batch in dl:
        input_data, src_data, target_data = batch
        pred = model(input_data, src_data)
        loss = criterion(pred / 2e-3, target_data)
        total_loss += loss.item()
        n_samples += input_data.shape[0]
    return total_loss / n_samples


def run_training(cfg, fold_id, total_df, speed_data, src_data, field_data):
    logging.info('================================================================')
    logging.info(f"==== Running training for fold {fold_id} ====")
    print('================================================================')
    print(f"==== Running training for fold {fold_id} ====")

    # == Data ==
    df_train = total_df[total_df['fold'] != fold_id].copy()
    df_valid = total_df[total_df['fold'] == fold_id].copy()

    logging.info(f"Train shape: {df_train.shape}")
    logging.info(f"Valid shape: {df_valid.shape}")
    print(f"Train shape: {df_train.shape}")
    print(f"Valid shape: {df_valid.shape}")

    dl_train, dl_val, ds_train, ds_val = get_fold_dls(cfg, df_train, df_valid, speed_data, src_data, field_data)

    # == model ==
    model = TrainingSystem(cfg)

    # == Load model ==
    if cfg['load_conf']['load']:
        model_ckpt = os.path.join(get_original_cwd(), 'outputs', cfg['load_conf']['folder'], f'fold_{fold_id}.ckpt')
        weights = torch.load(model_ckpt, map_location=model.device)['state_dict']
        model.load_state_dict(weights)

    # == Other component ==
    logger = TensorBoardLogger(save_dir=f"logs", default_hp_metric=False)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=f"./",
                                          save_top_k=1,
                                          save_last=False,
                                          save_weights_only=True,
                                          filename=f"fold_{fold_id}",
                                          mode='min')
    callbacks_to_use = [checkpoint_callback, TQDMProgressBar(refresh_rate=1)]

    # == Trainer ==
    trainer = pl.Trainer(
        max_epochs=cfg['train_conf']['epochs'],
        val_check_interval=cfg['train_conf']['val_interval'],
        callbacks=callbacks_to_use,
        logger=logger,
        enable_model_summary=False,
        accelerator="gpu",
        devices=cfg['device'],
        deterministic=True,
        precision='16-mixed' if cfg['mixed_precision'] else 32,
        strategy='ddp'
    )

    # == Training ==
    logging.info('Training the model')
    trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_val)

    best_model_path = checkpoint_callback.best_model_path
    logging.info('Predicting the validation set')
    logging.info(f"Best model path: {best_model_path}")

    # == Prediction ==
    # trainer.validate(model, dataloaders=dl_val, verbose=False, ckpt_path='best')

    val_score = model.val_score
    print(f'Fold {fold_id} RMSE: {val_score:.4f}')
    logging.info(f'Fold {fold_id} RMSE: {val_score:.4f}')
    return val_score


@hydra.main(version_base=None, config_path="./conf", config_name="train")
def main_fn(cfg):
    log_system = logging.getLogger(__name__)
    log_system.info(OmegaConf.to_yaml(cfg))

    # == global ==
    pl.seed_everything(cfg['seed'], workers=True)
    cfg['device'] = [eval(i) for i in str(cfg['device']).split('+')]

    # == data ==
    if cfg['data_conf']['v_dataset'] == 'v1':
        speed_data, src_data, field_data = load_data(cfg)
    elif cfg['data_conf']['v_dataset'] == 'v2':
        speed_data, src_data, field_data = load_data_v2(cfg)
    else:
        raise Exception('Dataset version not recognized from the list')

    print(f"Speed data shape: {speed_data.shape}")

    # crate metadata by filed_data
    if cfg['data_conf']['n_max'] == -1:
        metadata = {"sample_ID": list(range(speed_data.shape[0]))}
    else:
        assert cfg['data_conf']['n_max'] > 0
        metadata = {"sample_ID": list(range(min(cfg['data_conf']['n_max'], speed_data.shape[0])))}
    # to df
    metadata = pd.DataFrame(metadata)
    print(f'load {len(metadata)} different speed samples')
    logging.info(f'load {len(metadata)} different speed samples')

    # == Folds ==
    if cfg['load_conf']['load']:
        # load model, therefore load metadata from the same folder
        meta_path = os.path.join(get_original_cwd(), 'outputs', cfg['load_conf']['folder'], 'metadata.csv')
        print(f'load metadata.csv from: {meta_path}')
        metadata = pd.read_csv(meta_path)
    else:
        gkf = GroupKFold(n_splits=cfg['train_conf']['n_folds'])
        metadata['fold'] = 0
        for fold_id, (train_idx, val_idx) in enumerate(gkf.split(metadata, groups=metadata['sample_ID'])):
            metadata.loc[val_idx, 'fold'] = fold_id
        # save metadata
        metadata.to_csv('./metadata.csv', index=False)

    # == Test Dataset ==
    dummy_train = metadata[metadata['fold'] != 0].copy()
    dummy_valid = metadata[metadata['fold'] == 0].copy()

    dl_train, dl_val, ds_train, ds_val = get_fold_dls(cfg, dummy_train, dummy_valid, speed_data, src_data, field_data)

    show_batch(ds_train)

    # == Training ==
    torch.set_float32_matmul_precision('high')

    fold_val_score_list = list()

    for f in range(cfg['train_conf']['n_folds']):

        if cfg['train_conf']['only_first_fold']:
            if f != 0:
                break

        val_score = run_training(cfg, f, metadata, speed_data, src_data, field_data)

        fold_val_score_list.append(val_score)

    logging.info('==================================================')
    print('==================================================')

    for f, score in enumerate(fold_val_score_list):
        logging.info(f'Fold {f} RMSE: {score:.4f}')
        print(f'Fold {f} RMSE: {score:.4f}')

    mean_score = sum(fold_val_score_list) / len(fold_val_score_list)
    print(f'Mean RMSE: {mean_score:.4f}')
    logging.info(f'Mean RMSE: {mean_score:.4f}')

    return mean_score


if __name__ == '__main__':
    main_fn()
