import clip
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import shapenetcore_cat2id
from model.clip import TextPointCloudCLIP

shapenetcore_id2cat = {v: k for k, v in shapenetcore_cat2id.items()}

class LitTextPointCloudCLIP(L.LightningModule):
    def __init__(self, point_cloud_encoder, clip_name="ViT-B/32", device="cuda"):
        super().__init__()
        self.clip_model = TextPointCloudCLIP(point_cloud_encoder, clip_name, device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def _compute_logits(self, features_a, features_b):
        features_a = features_a / features_a.norm(dim=1, keepdim=True)
        features_b = features_b / features_b.norm(dim=1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits_per_a = logit_scale * features_a @ features_b.t()
        logits_per_b = logits_per_a.t()
        
        return logits_per_a, logits_per_b
        
    def _step(self, batch):
        pts_enc = batch["points_encoded"]
        labels = batch["label"].squeeze().tolist()
    
        text_features, point_cloud_features = self.clip_model(labels, pts_enc)
        
        logits_per_text, logits_per_point_cloud = self._compute_logits(
            text_features, point_cloud_features)
        
        ground_truth = torch.arange(len(labels)).to(self.device)
        clip_loss = (self.loss_fn(logits_per_text, ground_truth) + self.loss_fn(logits_per_point_cloud, ground_truth)) / 2
        
        # Calculate accuracy
        pred_text = logits_per_text.argmax(dim=1)
        pred_point_cloud = logits_per_point_cloud.argmax(dim=1)
        accuracy_text = (pred_text == ground_truth).float().mean()
        accuracy_point_cloud = (pred_point_cloud == ground_truth).float().mean()
        accuracy = (accuracy_text + accuracy_point_cloud) / 2

        return clip_loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self._step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss, "accuracy": accuracy}

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "accuracy": accuracy}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-6)
        return optimizer