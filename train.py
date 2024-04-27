import os
import hydra
import logging

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from avg_ckpts import ensemble
from datamodule.data_module import DataModule
from pytorch_lightning.loggers import TensorBoardLogger

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    seed_everything(42, workers=True)
    cfg.gpus = torch.cuda.device_count()

    checkpoint = ModelCheckpoint(
        monitor="monitoring_step",
        mode="max",
        dirpath=os.path.join(cfg.exp_dir, cfg.exp_name) if cfg.exp_dir else None,
        save_last=True,
        filename="{epoch}",
        save_top_k=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    # Set modules and trainer
    if cfg.data.modality in ["audio", "visual"]:
        from lightning import ModelModule
    elif cfg.data.modality == "audiovisual":
        from lightning_av import ModelModule
    modelmodule = ModelModule(cfg)
    datamodule = DataModule(cfg)
    modelmodule.model.load_state_dict(torch.load("./avsr_trlrwlrs2lrs3vox2avsp_base.pth", map_location=lambda storage, loc: storage))

    for param in modelmodule.model.parameters():
        param.requires_grad = False
    for param in modelmodule.model.encoder.parameters():
        param.requires_grad = True

    trainer = Trainer(
        **cfg.trainer,
        log_every_n_steps=1,
        logger=TensorBoardLogger("tb_transfer_learning_not_rt", name="auto_avsr", default_hp_metric=False),
        callbacks=callbacks,
        strategy=DDPPlugin(find_unused_parameters=False)
    )

    trainer.fit(model=modelmodule, datamodule=datamodule)
    ensemble(cfg)


if __name__ == "__main__":
    main()
