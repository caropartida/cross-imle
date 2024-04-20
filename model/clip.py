import clip
import torch

from dataset import shapenetcore_cat2id

shapenetcore_id2cat = {v: k for k, v in shapenetcore_cat2id.items()}

# CLIP Model
##############################################################################
class TextPointCloudCLIP(torch.nn.Module):
    def __init__(self, point_cloud_encoder, clip_name="ViT-B/32", device="cuda"):
        super().__init__()
        self.pretrained_model, self.preprocess = clip.load(clip_name, device=device, jit=False)
        self.pretrained_model = self.pretrained_model.float()
        self.pretrained_model.eval()
        self.point_cloud_encoder = point_cloud_encoder
        
        label_list = [i for i in range(len(shapenetcore_cat2id))]
        text = clip.tokenize([
            shapenetcore_id2cat[l] for l in labels
        ]).to(self.device)
        with torch.no_grad():
            self.label_latents = self.pretrained_model.encode_text(text).detach()
        
        assert self.point_cloud_encoder.training == True
        
    def forward(self, labels, encoded_points):        
        text_features = self.compute_label_features(labels)
        point_cloud_features = self.point_cloud_encoder(encoded_points)
        return text_features, point_cloud_features
    
    def compute_label_features(self, labels):
        text_features = torch.stack([self.label_latents[i] for i in labels]).squeeze(-1)
        text_features = self.dim_reduction(text_features)
        return text_features
    