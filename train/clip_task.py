import datetime
import os
import sys
from types import SimpleNamespace

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from dataset import AEDataset, RobustAEDataset
from lightning_model.autoencoder import LitAE
from lightning_model.clip import LitTextPointCloudCLIP

from models.dgcnn import DGCNN, ResNet

config = {
    'enc_filters': (64, 128, 128, 256),
    'latent_dim': 128,
    'enc_bn': True,
    'dec_features': (256, 256),
    'n_pts': 256,
    'dec_bn': False,
}
config = SimpleNamespace(**config)

checkpoint = 'lightning_logs/ae_model_20240410-103850/version_0/checkpoints/epoch=472-step=84667.ckpt'
autoencoder = LitAE.load_from_checkpoint(checkpoint, config=config)
encoder = autoencoder.autoencoder.encoder

cross_model = DGCNN()
cross_model.load_state_dict(torch.load('./checkpoints/dgcnn_cls_best.pth'))
point_model = cross_model.to(device)

lit_clip_model = LitTextPointCloudCLIP(encoder, clip_name="ViT-B/32", device='cuda:0')

root = ''
dataset_name = 'shapenetcorev2'
# choose split type from 'train', 'test', 'all', 'trainval' and 'val'
# only shapenetcorev2 and shapenetpart dataset support 'trainval' and 'val'

train_dataset = RobustAEDataset(root=root, dataset_name=dataset_name, split='train') # CLIP on random point cloud
val_dataset = RobustAEDataset(root=root, dataset_name=dataset_name,  split='val') #  num_sample_points=config.n_pts,
# test_dataset = AEDataset(root=root, dataset_name=dataset_name, num_sample_points=config.n_pts, split='test')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
# test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False, num_workers=4)
# print("datasize:", train_dataset.__len__())
# print("datasize:", val_dataset.__len__())
# print("datasize:", test_dataset.__len__())


checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")
logger = TensorBoardLogger(save_dir=f'lightning_logs', name=f"clip_model")
trainer = L.Trainer(max_epochs=5000, gpus=1, logger=logger, callbacks=[checkpoint_callback])
trainer.fit(model=lit_clip_model, train_dataloaders=train_loader, val_dataloaders=val_loader)