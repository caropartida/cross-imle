import sys

sys.path.append('../')
import datetime
from types import SimpleNamespace

import clip
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader

from dataset import AEDataset, RobustAEDataset, shapenetcore_cat2id
from lightning_model.autoencoder import LitAE
from lightning_model.clip import LitTextPointCloudCLIP
from lightning_model.imle import LitIMLEGenerator

from models.dgcnn import DGCNN, ResNet
import argparse

shapenetcore_id2cat = {v: k for k, v in shapenetcore_cat2id.items()}

config = {
    'enc_filters': (64, 128, 128, 256),
    'latent_dim': 128,
    'enc_bn': True,
    'dec_features': (256, 256),
    'n_pts': 256,
    'dec_bn': False,
    'noise_dim': 32,
    'num_latent': 80,
    'imle_features': (256, 512),
    'latent_loss_weight': 1000,
}
config = SimpleNamespace(**config)
checkpoint = 'lightning_logs/ae_model_20240410-103850/version_0/checkpoints/epoch=472-step=84667.ckpt'
autoencoder = LitAE.load_from_checkpoint(checkpoint, config=config)

# actually can we just use Tony's autoencoder no?
# point_model = DGCNN(args)
# point_model.load_state_dict(torch.load('./checkpoints/dgcnn_cls_best.pth'))
# point_model = point_model.to(device)
# img_model = ResNet(resnet50(), feat_dim = 256)
# img_model = img_model.to(device)

# checkpoint = 'lightning_logs/clip_model_20240410-200506/version_0/checkpoints/epoch=93-step=104904.ckpt'
# clip_model = LitTextPointCloudCLIP.load_from_checkpoint(checkpoint, config=config, point_cloud_encoder=autoencoder.autoencoder.encoder)
autoencoder.eval().cuda()
# clip_model.eval().cuda()

root = ''
dataset_name = 'shapenetcorev2'
# choose split type from 'train', 'test', 'all', 'trainval' and 'val'
# only shapenetcorev2 and shapenetpart dataset support 'trainval' and 'val'

train_dataset = RobustAEDataset(root=root, dataset_name=dataset_name, split='train')
val_dataset = RobustAEDataset(root=root, dataset_name=dataset_name, split='val')
# test_dataset = AEDataset(root=root, dataset_name=dataset_name, num_sample_points=config.n_pts, split='test')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
# print("datasize:", train_dataset.__len__())
# print("datasize:", val_dataset.__len__())
# print("datasize:", test_dataset.__len__())


parameters = list(point_model.parameters()) + list(img_model.parameters())

pretrained_model, preprocess = clip.load("ViT-B/32", device='cuda', jit=False)

label_list = list(shapenetcore_id2cat.values())
text = clip.tokenize([
    l for l in label_list
]).to('cuda')
with torch.no_grad():
    label_latents = pretrained_model.encode_text(text).detach().float()
    
lit_lmle_model = LitIMLEGenerator(
    config, label_latents, autoencoder
)

checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")
logger = TensorBoardLogger(save_dir=f'lightning_logs/imle')
trainer = L.Trainer(max_epochs=5000, gpus=1, logger=logger, callbacks=[checkpoint_callback]) 
trainer.fit(model=lit_lmle_model, train_dataloaders=train_loader, val_dataloaders=val_loader)