import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# Import from MOVIAD
from moviad.models.vad_model import VADModel
from moviad.models.training_args import TrainingArgs
from moviad.common.common_losses import FocalLoss, BinaryDiceLoss

# Import from AnomalyCLIP
from . import AnomalyCLIP_lib
from .prompt_ensemble import AnomalyCLIP_PromptLearner

class AnomalyCLIPArgs(TrainingArgs):

    def init_train(self, model):
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                list(model.prompt_learner.parameters()),
                lr=0.001,
                betas=(0.5, 0.999)
            )
        
    def __to_dict__(self):
        return {
            "optimizer": self.optimizer_to_dict(self.optimizer) if hasattr(self, 'optimizer') and self.optimizer else None,
        }

class AnomalyCLIP(VADModel):

    def __init__(
        self,
        device: torch.device,
        features_list: list = [6, 12, 18, 24],
        feature_map_layer: list = [0, 1, 2, 3],
        dpam_layer: int = 20,
        n_ctx: int = 12,
        depth: int = 9,
        t_n_ctx: int = 4,
        pretrained_model: str = "ViT-L/14@336px",
        checkpoint: str | None = None,
        sigma: int = 4,
        alpha: float = 0.5  # Weight to balance Zero-shot and Few-shot branches
    ):
        super().__init__()
        
        self.device = device
        self.features_list = features_list
        self.feature_map_layer = feature_map_layer
        self.dpam_layer = dpam_layer
        self.sigma = sigma
        self.alpha = alpha
        
        # Memory Bank Initialization
        self.mem_image_features = None  # Will store global features [K, C]
        self.mem_patch_features = None  # Will store patch features [K, Layers, P, C]

        # AnomalyCLIP parameters
        anomalyclip_params = {
            "Prompt_length": n_ctx,
            "learnabel_text_embedding_depth": depth,
            "learnabel_text_embedding_length": t_n_ctx
        }
        
        # Load pretrained CLIP model
        self.model, _ = AnomalyCLIP_lib.load(
            pretrained_model, 
            device=device, 
            design_details=anomalyclip_params
        )
        self.model.eval()
        
        # Initialize prompt learner
        self.prompt_learner = AnomalyCLIP_PromptLearner(
            self.model.to("cpu"), 
            anomalyclip_params
        )

        if checkpoint is not None:
            checkpoint_path = AnomalyCLIP_lib.download_prompt_learner(checkpoint)
            self.load(checkpoint_path)

        self.prompt_learner.to(device)
        self.model.to(device)
        
        # Apply DPAM to vision encoder
        self.model.visual.DAPM_replace(DPAM_layer=dpam_layer)
        
        # Loss functions
        self.loss_focal = FocalLoss()
        self.loss_dice = BinaryDiceLoss()
        self.lam = 4  
        
        self.model.eval()

    def to(self, device: torch.device):
        super().to(device)
        self.model.to(device)
        self.prompt_learner.to(device)
        if self.mem_image_features is not None:
            self.mem_image_features = self.mem_image_features.to(device)
        if self.mem_patch_features is not None:
            self.mem_patch_features = [f.to(device) for f in self.mem_patch_features]
        self.device = device
        return self

    def build_memory_bank(self, normal_dataloader):
        """
        Populates the memory bank by extracting features from the provided k normal images.
        """
        self.model.eval()
        image_features_list = []
        patch_features_list = []

        print("-> Populating Memory Bank with normal images...")
        with torch.no_grad():
            for batch in normal_dataloader:

                images = batch
                
                images = images.to(self.device)
                
                img_feats, patch_feats = self.model.encode_image(
                    images, 
                    self.features_list, 
                    DPAM_layer=self.dpam_layer
                )
                
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                image_features_list.append(img_feats)
                patch_features_list.append(patch_feats)

        # Concatenate extracted features
        self.mem_image_features = torch.cat(image_features_list, dim=0) # [K, C]
        
        self.mem_patch_features = []
        num_layers = len(patch_features_list[0])
        for l_idx in range(num_layers):
            layer_patches = torch.cat([batch_patches[l_idx] for batch_patches in patch_features_list], dim=0)
            layer_patches = layer_patches / layer_patches.norm(dim=-1, keepdim=True)
            self.mem_patch_features.append(layer_patches) # List of [K, Num_Patches, C]

        print(f"-> Memory Bank successfully completed ({self.mem_image_features.shape[0]} images).")

    def forward(self, images: torch.Tensor, **kwargs):
        
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        
        self.image_size = images.shape[2]

        # 1. Query Feature Extraction from Vision Encoder
        with torch.no_grad():
            image_features, patch_features = self.model.encode_image(
                images, 
                self.features_list, 
                DPAM_layer=self.dpam_layer
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 2. Process learned textual prompts
        prompts, tokenized_prompts, compound_prompts_text = self.prompt_learner(cls_id=None)
        text_features = self.model.encode_text_learn(
            prompts, 
            tokenized_prompts, 
            compound_prompts_text
        ).float()
        
        text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Score based purely on prompts
        text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
        text_probs = text_probs[:, 0, ...] / 0.07

        # 3. Multiscale Similarity Maps Generation
        similarity_map_list = []
        for idx, patch_feature in enumerate(patch_features):
            if idx >= self.feature_map_layer[0]:
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                
                similarity, _ = AnomalyCLIP_lib.compute_similarity(
                    patch_feature, 
                    text_features[0]
                )
                similarity_map = AnomalyCLIP_lib.get_similarity_map(
                    similarity[:, 1:, :], 
                    self.image_size
                ).permute(0, 3, 1, 2)
                
                # --- Few-Shot Integration on Patch Map ---
                if self.mem_patch_features is not None and not self.prompt_learner.training:

                    mem_patch = self.mem_patch_features[idx]
                    
                    B, P, C = patch_feature.shape
                    K, _, _ = mem_patch.shape

                    # Isolate only the patch component by excluding the CLS token (patch_feature[:, 1:, :])
                    q_patches = patch_feature[:, 1:, :] # [B, P-1, C]
                    m_patches = mem_patch[:, 1:, :].reshape(-1, C) # [K * (P-1), C]
                    
                    # Coupling matrix [B, P-1, K*(P-1)]
                    patch_sim = torch.matmul(q_patches, m_patches.t())

                    # Get top-k similarity
                    topk_sim, _ = torch.topk(patch_sim, k=5, dim=-1)
                    avg_topk_sim = topk_sim.mean(dim=-1)
                    
                    # Map Cosine Similarity from [-1.0, 1.0] to [0.0, 1.0]
                    avg_topk_sim_scaled = (avg_topk_sim + 1.0) / 2.0
                    
                    few_shot_anomaly = 1.0 - avg_topk_sim_scaled
                    
                    # Explicit spatial reshape, without assuming it matches compute_similarity output structure
                    side = int(few_shot_anomaly.shape[1] ** 0.5)
                    fs_map = few_shot_anomaly.reshape(B, side, side).unsqueeze(1)  # [B, 1, H, W]
                    fs_map = F.interpolate(fs_map, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
                    # fs_map: [B, 1, H, W] — pure few-shot anomaly map
                    
                    # similarity_map[:, 1, :, :] is the anomaly channel of the textual branch [B, H, W]
                    similarity_map[:, 1, :, :] = (1 - self.alpha) * similarity_map[:, 1, :, :] + self.alpha * fs_map.squeeze(1)
                    similarity_map[:, 0, :, :] = 1 - similarity_map[:, 1, :, :]
                
                similarity_map_list.append(similarity_map)
        
        if self.prompt_learner.training:
            return text_probs, similarity_map_list
        
        else:
            anomaly_scores = F.softmax(text_probs, dim=-1)[:, 1]

            # --- Few-Shot Integration on Global Score ---
            if self.mem_image_features is not None:

                memory_sim = torch.matmul(image_features, self.mem_image_features.t()) # [B, K]
                avg_normal_sim = torch.mean(memory_sim, dim=-1) # [B]
                
                # Map Similarity from [-1.0, 1.0] to [0.0, 1.0]
                avg_normal_sim_scaled = (avg_normal_sim + 1.0) / 2.0
                
                few_shot_score = 1.0 - avg_normal_sim_scaled
                
                anomaly_scores = (1 - self.alpha) * anomaly_scores + self.alpha * few_shot_score

            # Final aggregation of anomaly maps
            if len(similarity_map_list) > 0:
                anomaly_maps_scaled = []
                for similarity_map in similarity_map_list:
                    am = (similarity_map[:, 1, :, :] + 1 - similarity_map[:, 0, :, :]) / 2.0
                    anomaly_maps_scaled.append(am)
                
                anomaly_map = torch.stack(anomaly_maps_scaled).sum(dim=0)
                anomaly_map = torch.stack([
                    torch.from_numpy(gaussian_filter(i, sigma=self.sigma)) 
                    for i in anomaly_map.detach().cpu()
                ], dim=0)
            else:
                anomaly_map = torch.zeros(images.shape[0], self.image_size, self.image_size, device=self.device)
        
            return anomaly_map, anomaly_scores


    def train_step(self, batch: torch.Tensor, training_args: TrainingArgs):

        image = batch[0].to(self.device)
        label = batch[1].to(self.device)
        gt = batch[2].to(self.device)
        
        gt[gt > 0.5] = 1
        gt[gt <= 0.5] = 0
        
        text_probs, similarity_map_list = self(image)

        if text_probs.shape[1] == 1:
            text_probs = text_probs.squeeze(1)

        image_loss = F.cross_entropy(text_probs, label.long())
           
        seg_loss = 0
        for similarity_map in similarity_map_list:
            seg_loss += self.loss_focal(similarity_map, gt)
            seg_loss += self.loss_dice(similarity_map[:, 1, :, :], gt)
            seg_loss += self.loss_dice(similarity_map[:, 0, :, :], 1 - gt)
        
        total_loss = self.lam * seg_loss + image_loss
        
        training_args.optimizer.zero_grad()
        total_loss.backward()
        training_args.optimizer.step()
        
        return total_loss.item()
    
    def train_epoch(self, epoch: int, train_dataloader, training_args: TrainingArgs):

        self.model.eval()
        self.prompt_learner.train()
        avg_batch_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            loss = self.train_step(batch, training_args)
            avg_batch_loss += loss
        return avg_batch_loss / len(train_dataloader)

    def reset_model(self):
        anomalyclip_params = {"Prompt_length": 12, "learnabel_text_embedding_depth": 9, "learnabel_text_embedding_length": 4}
        self.prompt_learner = AnomalyCLIP_PromptLearner(self.model.to("cpu"), anomalyclip_params)
        self.prompt_learner.to(self.device)
        self.model.to(self.device)
        self.mem_image_features = None
        self.mem_patch_features = None

    def save_model(self, save_path: str):
        torch.save({"prompt_learner": self.prompt_learner.state_dict()}, save_path)
    
    def load(self, load_path: str):
        checkpoint = torch.load(load_path, map_location=torch.device("cpu"))
        self.prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    
    def get_model_size(self):
        prompt_learner_params = sum(p.numel() for p in self.prompt_learner.parameters())
        clip_params = sum(p.numel() for p in self.model.parameters())
        return {
            "prompt_learner_params": prompt_learner_params,
            "clip_model_params": clip_params,
            "total_params": prompt_learner_params + clip_params,
            "trainable_params": prompt_learner_params
        }