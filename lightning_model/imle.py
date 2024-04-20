import contextlib
import time

import lightning as L
import torch
from torch import nn, optim

from dciknn_cuda import DCI
from emd import earth_mover_distance
from model.imle_gen import IMLEGenerator
from utils import local_directed_hausdorff


class DCI_Helper:
    # def __init__(self):
        # self.dci_db = DCI(128,2,10,100,10)
        
    def __call__(self, x, y):        
        # self.dci_db.add(x)
        # index, _ = self.dci_db.query(y)
        # val = index[0][0].long().item()
        # self.dci_db.clear()
        # return val
        
        distances = ((x - y) ** 2).sum(1)
        # Find the index of the nearest neighbor
        min_index = distances.argmin()
        return min_index



class LitIMLEGenerator(L.LightningModule):
    def __init__(self, config, label_latents, autoencoder):
        super().__init__()
        
        self.label_latents = label_latents
        self.autoencoder = autoencoder
        self.autoencoder.eval()
        
        self.imle_gen = IMLEGenerator(
            latent_dim=config.latent_dim,
            noise_dim=config.noise_dim,
            n_features=config.imle_features,
            num_latent=config.num_latent
        ).cuda()

        self.dci_db = DCI_Helper()
        self.mse_loss = nn.MSELoss()

        self.latent_loss_weight = config.latent_loss_weight

    def find_closest_latent(self, generated_pc_latent, real_pc_latent):
        selected_generated_pc_latent = []
        
        n_samples = generated_pc_latent.shape[0]
        
        for i in range(n_samples):
            generated_pc_latent_i = generated_pc_latent[i] # [num_latent, latent_dim] 
            real_pc_latent_i = real_pc_latent[i].unsqueeze(0) # [1, latent_dim]
            
            index = self.dci_db(generated_pc_latent_i, real_pc_latent_i)
            
            selected_generated_pc_latent.append(generated_pc_latent_i[index])
            
        selected_generated_pc_latent = torch.stack(selected_generated_pc_latent)
        
        torch.cuda.empty_cache()
        return selected_generated_pc_latent

    def forward(self, batch, batch_idx):
        label_list = batch['label'].squeeze(-1).tolist()
        real_pc = batch['points'].cuda()
        real_pc_enc = batch['points_encoded'].cuda()                                     
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        with torch.no_grad():
            label_latent = torch.stack([self.label_latents[i] for i in label_list]).squeeze(-1).cuda()
            real_pc_latent = self.autoencoder.autoencoder.encoder(real_pc_enc)
            
        generated_pc_latent = self.imle_gen(label_latent, real_pc_latent)
        
        # # This is some how required to avoid DCI from crashing
        # real_pc_latent = torch.empty(real_pc_latent.shape).copy_(real_pc_latent).detach().cuda()
                
        selected_generated_pc_latent = self.find_closest_latent(
            generated_pc_latent, real_pc_latent)
        
        with torch.no_grad():
            generated_pc = self.autoencoder.autoencoder.decoder(selected_generated_pc_latent)    
            
        return selected_generated_pc_latent, real_pc_latent, generated_pc, real_pc


    def _step(self, batch, batch_idx):
        selected_generated_pc_latent, real_pc_latent, generated_pc, real_pc = self(batch, batch_idx)

        latent_loss = self.mse_loss(selected_generated_pc_latent, real_pc_latent) * self.latent_loss_weight
        
        emd_loss = earth_mover_distance(generated_pc, real_pc)
        emd_loss = emd_loss.mean()
        # local_directed_hausdorff(
        #     real_pc, generated_pc
        # ) * self.shape_loss_weight
        
        return latent_loss, emd_loss
    
    def generate(self, category_i):
        with torch.no_grad():
            label_latent = self.imle_gen.dim_reduction(self.label_latents[category_i].unsqueeze(0))
            noise = self.imle_gen.noise_generator.sample([1, self.imle_gen.noise_dim]).cuda()
            generated_pc_latent = self.imle_gen.generator(
                torch.cat([label_latent, noise], dim=1)
            )
            generated_pc = self.autoencoder.autoencoder.decoder(generated_pc_latent)
        return generated_pc
    
    def training_step(self, batch, batch_idx):
        latent_loss, emd_loss = self._step(batch, batch_idx)
        loss = latent_loss 
        self.log('emd_loss', emd_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('latent_loss', latent_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        latent_loss, emd_loss= self._step(batch, batch_idx)
        loss = latent_loss 
        self.log('val_emd_loss', emd_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_latent_loss', latent_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6)
        return optimizer