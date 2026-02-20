from typing import List, Tuple, Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.vad_model import VADModel
from moviad.models.training_args import TrainingArgs
from moviad.models.stfpm.loss_functions import stfpm_loss

@dataclass
class STFPMTrainArgs(TrainingArgs):

    def init_train(self, model: VADModel):
        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(
                #model.parameters(), lr=0.4, weight_decay=1e-4, momentum=0.9
                #model.parameters(), lr=0.04, weight_decay=1e-4, momentum=0.9
                #model.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9
                model.parameters(), lr=self.lr, weight_decay=1e-4, momentum=0.9
            )
        if self.loss_function is None:
            self.loss_function = stfpm_loss


class STFPM(VADModel):
    def __init__(
        self,
        teacher: CustomFeatureExtractor,
        student: CustomFeatureExtractor,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.device = torch.device("cpu")
        self.mse_loss = torch.nn.MSELoss(reduction="sum")

    def forward(self, batch: torch.Tensor):
        if self.training:
            teacher_features, student_features = None, None
            with torch.no_grad():
                teacher_features = self.teacher(batch)
            student_features = self.student(batch)

            return teacher_features, student_features
        else:
            student_features = self.student(batch)
            teacher_features = self.teacher(batch) 

            return self.post_process(
                teacher_features, student_features, batch.shape[2:]
            )
        
    def to(self, device: torch.device):
        super().to(device)
        self.teacher.to(device)
        self.student.to(device)
        self.device = device
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.model.eval()
        self.student.model.train(mode)
        return self

    def parameters(self):
        return self.student.model.parameters()

    def train_epoch(
        self, epoch, train_dataloader, training_args: STFPMTrainArgs
    ):
        avg_batch_loss = 0

        # train the model
        for batch in tqdm(train_dataloader):
            avg_batch_loss += self.train_step(batch, training_args)

        avg_batch_loss /= len(train_dataloader)
        return avg_batch_loss

    def train_step(self, batch: torch.Tensor, training_args: TrainingArgs):
        batch = batch.to(self.device)
        teacher_features, student_features = self.forward(batch)

        loss = 0

        for i in range(len(student_features)):
            teacher_features[i] = F.normalize(teacher_features[i], dim=1)
            student_features[i] = F.normalize(student_features[i], dim=1)
            #loss += training_args.loss_function(teacher_features[i], student_features[i])
            height, width = teacher_features[i].shape[2:]
            loss += (0.5 / (width * height)) * self.mse_loss(teacher_features[i], student_features[i])

        training_args.optimizer.zero_grad()
        loss.backward()
        training_args.optimizer.step()

        return loss.item()

    def post_process(self, t_feat, s_feat, output_shape) -> torch.Tensor:
        """
        This method actually produces the anomaly maps for evalution purposes

        Args:
            - t_feat: teacher features maps
            - s_feat: student features maps

        Returns:
            - anomaly maps

        """

        device = t_feat[0].device
        score_maps = torch.tensor([1.0], device=device)
        for j in range(len(t_feat)):
            t_feat[j] = F.normalize(t_feat[j], dim=1)
            s_feat[j] = F.normalize(s_feat[j], dim=1)
            sm = torch.sum((t_feat[j] - s_feat[j]) ** 2, 1, keepdim=True)
            sm = F.interpolate(
                sm, size=output_shape, mode="bilinear", align_corners=False
            )
            # aggregate score map by element-wise product
            score_maps = score_maps * sm

        anomaly_scores = torch.max(score_maps.view(score_maps.size(0), -1), dim=1)[0]
        return score_maps, anomaly_scores
