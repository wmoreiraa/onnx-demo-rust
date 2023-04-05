
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

class Model:
    def __init__(self, data_modules: dict):
        self.data_modules = data_modules

    def run(self) -> TemporalFusionTransformer:
                # configure network and trainer
        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()  # log the learning rate
        logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

        trainer = pl.Trainer(
            max_epochs=30,
            gpus=1,
            enable_model_summary=True,
            gradient_clip_val=0.1,
            #limit_train_batches=30,  # coment in for training, running valiati
            #fast_dev_run=True,  # comment in to check that networkor datase
            callbacks=[lr_logger, early_stop_callback],
            logger=logger,
        )


        tft = TemporalFusionTransformer.from_dataset(
            self.data_modules["training"],
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7,  # 7 quantiles by default
            loss=QuantileLoss(),
            log_interval=10,  
            reduce_on_plateau_patience=4,
        )
        print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
        
        trainer.fit(
            tft,
            train_dataloaders=self.data_modules["train_dataloader"],
            val_dataloaders=self.data_modules["val_dataloader"])
        
        trainer.save_checkpoint("tft.ckpt")
