import datetime
from types import SimpleNamespace

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from dataset import AEDataset, RobustAEDataset
from lightning_model.autoencoder import LitAE

config = {
    'enc_filters': (64, 128, 128, 256),
    'latent_dim': 128,
    'enc_bn': True,
    'dec_features': (256, 256),
    'n_pts': 256,
    'dec_bn': False,
    'robust_sampling': False,
}
config = SimpleNamespace(**config)

root = '.'
dataset_name = 'shapenetcorev2'
# choose split type from 'train', 'test', 'all', 'trainval' and 'val'
# only shapenetcorev2 and shapenetpart dataset support 'trainval' and 'val'

if config.robust_sampling:
    train_dataset = RobustAEDataset(root=root, dataset_name=dataset_name, split='train')
else:
    train_dataset = AEDataset(root=root, dataset_name=dataset_name, num_sample_points=config.n_pts, split='train')
    
val_dataset = AEDataset(root=root, dataset_name=dataset_name, num_sample_points=config.n_pts, split='val')
test_dataset = AEDataset(root=root, dataset_name=dataset_name, num_sample_points=config.n_pts, split='test')

train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=200, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
# print("datasize:", train_dataset.__len__())
# print("datasize:", val_dataset.__len__())
# print("datasize:", test_dataset.__len__() 

autoencoder = LitAE(config)
lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")
logger = TensorBoardLogger(save_dir=f'lightning_logs', name=f"ae_model")
# TODO Change max epoch to 1000
trainer = L.Trainer(max_epochs=10000, gpus=1, logger=logger, callbacks=[checkpoint_callback, lr_monitor])



# import cProfile
# import pstats

# profiler = cProfile.Profile()
# profiler.enable()

trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(model=autoencoder, test_dataloaders=test_loader)

# # Disable profiling
# profiler.disable()
# # Create a Stats object based on the information inside the profiler
# stats = pstats.Stats(profiler)
# # Sort the statistics by the cumulative time spent in the function
# stats.sort_stats('cumulative')
# # Print the stats to the console
# stats.print_stats()
