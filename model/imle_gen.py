import torch
import torch.distributions as normal
import torch.nn as nn


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		m.weight.data.normal_(0.0, 0.02)
		m.bias.data.fill_(0)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

class IMLEGenerator(torch.nn.Module):
    def __init__(self, latent_dim=128, noise_dim=32, n_features=(256, 512), num_latent=80):
        super(IMLEGenerator, self).__init__()
        self.num_latent = num_latent
        self.latent_dim = latent_dim
        self.n_features = n_features
        self.noise_dim = noise_dim
    
        self.noise_generator = normal.Normal(0, 4)
    
        self.generator = self._create_model()
        self.apply(weights_init)
        
        self.dim_reduction = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 128)
        )
    
    def _create_model(self):
        model = []
        prev_nf = 128 + self.noise_dim 
        
        for nf in self.n_features:
            model.append(torch.nn.Linear(prev_nf, nf))
            model.append(torch.nn.LeakyReLU(inplace=True))
            prev_nf = nf
            
        model.append(torch.nn.Linear(prev_nf, self.latent_dim))
        return nn.Sequential(*model)

    def forward(self, label_latent, real_pc_latent):
        generated_pc_latent_list = []
        label_latent = self.dim_reduction(label_latent)
        
        for i in range(self.num_latent):
            noise = self.noise_generator.sample([
                real_pc_latent.shape[0], self.noise_dim
            ]).cuda()
            
            generated_pc_latent = self.generator(
                torch.cat([label_latent, noise], dim=1)
            )
            generated_pc_latent_list.append(generated_pc_latent)
            
        return torch.stack(generated_pc_latent_list, 1) #[B, num_latent, latent_dim]