from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from tqdm import tqdm

from moviad.models.components.rd4ad.deresnet import de_resnet18, de_wide_resnet50_2
from moviad.models.components.rd4ad.resnet import resnet18, wide_resnet50_2
from moviad.models.rd4ad.loss_functions import rd4ad_cosine_loss
from moviad.models.training_args import TrainingArgs
from moviad.models.vad_model import VADModel


@dataclass
class RD4ADTrainArgs(TrainingArgs):
    def init_train(self, model: VADModel):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                list(model.decoder.parameters()) + list(model.bn.parameters()),
                lr=0.005,
                betas=(0.5, 0.999),
            )
        if self.loss_function is None:
            self.loss_function = rd4ad_cosine_loss


class RD4AD(VADModel):
    DEFAULT_PARAMETERS = {
        "epochs": 200,
        "batch_size": 16,
        "learning_rate": 0.005,
        "betas": (0.5, 0.999),
    }

    MAPPING = {
        "resnet18": {
            "trainable": partial(resnet18, pretrained=True),
            "non_trainable": partial(de_resnet18, pretrained=False),
        },
        "wide_resnet50_2": {
            "trainable": partial(wide_resnet50_2, pretrained=True),
            "non_trainable": partial(de_wide_resnet50_2, pretrained=False),
        },
    }

    def __init__(self, backbone_name, input_size=(224, 224)):
        super().__init__()

        self.device = torch.device("cpu")
        self.input_size = input_size

        if backbone_name in self.MAPPING:
            models = self.MAPPING[backbone_name]

            self.encoder, self.bn = models["trainable"]()
            self.decoder = models["non_trainable"]()

            # freeze the pretrained encoder (teacher)
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def to(self, device: torch.device):
        self.encoder.to(device)
        self.bn.to(device)
        self.decoder.to(device)
        self.device = device
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        self.bn.train(mode)
        self.decoder.train(mode)
        return self

    def forward(self, batch: torch.Tensor):
        """
        Output tensors
        List[torch.Tensor] of len (n_layers)
        every tensor shape is (B C H W)
        """
        enc_batch = self.encoder(batch)
        bn_batch = self.bn(enc_batch)
        dec_batch = self.decoder(bn_batch)

        if self.training:
            return enc_batch, bn_batch, dec_batch
        else:
            return self.post_process(enc_batch, dec_batch)

    def train_epoch(self, epoch, train_dataloader, training_args: RD4ADTrainArgs):
        avg_batch_loss = 0

        # train the model
        for batch in tqdm(train_dataloader):
            avg_batch_loss += self.train_step(batch, training_args)

        avg_batch_loss /= len(train_dataloader)
        return avg_batch_loss

    def compute_loss(self, batch: torch.Tensor, training_args: TrainingArgs) -> torch.Tensor:
        batch = batch.to(self.device)
        teacher_features, _, student_features = self.forward(batch)

        loss = 0

        for i in range(len(student_features)):
            loss += training_args.loss_function(
                teacher_features[i], student_features[i]
            )

        return loss

    def train_step(self, batch: torch.Tensor, training_args: TrainingArgs):
        loss = self.compute_loss(batch, training_args)

        training_args.optimizer.zero_grad()
        loss.backward()
        training_args.optimizer.step()

        return loss.item()

    def post_process(self, enc_batch, dec_batch) -> torch.Tensor:
        anomaly_map = None
        sigma = 4
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        blur = GaussianBlur(kernel_size=kernel_size, sigma=4)

        # iterate over the feature extraction layers batches
        for i in range(len(enc_batch)):
            fs = dec_batch[i]
            ft = enc_batch[i]

            a_map = 1 - F.cosine_similarity(fs, ft)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(
                a_map,
                size=self.input_size,
                mode="bilinear",
                align_corners=True,
            )

            if anomaly_map is None:
                anomaly_map = a_map
            else:
                anomaly_map += a_map

        anomaly_map = blur(anomaly_map)
        return anomaly_map, torch.max(anomaly_map.view(anomaly_map.size(0), -1), dim=1)[0]

    def __call__(self, batch):
        return self.forward(batch)
