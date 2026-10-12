from moviad.datasets.vad_dataset import VADDataset
from moviad.scenarios.continual.continual_model import ContinualModel
from moviad.models.patchcore.patchcore import PatchCore
from moviad.models.training_args import TrainingArgs

import torch
from tqdm import tqdm

class PatchCoreFineTuningPP(ContinualModel):

    def start_task(self, task_index: int, train_dataset: VADDataset, train_args: TrainingArgs = None):
        pass    

    def train_task(self, task_index: int, train_dataset, eval_dataset, metrics, device, logger = None, train_args:TrainingArgs = None):

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_args.batch_size,
            shuffle=True,
            num_workers=train_args.num_workers,
        )

        embeddings = []

        with torch.no_grad():

            print("Embedding Extraction:")
            for batch in tqdm(iter(train_dataloader)):
                embedding = self.vad_model(batch)
                embeddings.append(embedding)

            embeddings = torch.cat(embeddings, dim = 0)
            torch.cuda.empty_cache()

            if task_index > 0:
                embeddings = torch.cat([embeddings, self.vad_model.memory_bank], dim=0)

            #apply coreset reduction
            print("Coreset Extraction:")
            coreset = self.vad_model.coreset_extractor.extract_coreset(embeddings)

            self.vad_model.memory_bank = coreset


    def end_task(self, task_index:int, train_dataset: VADDataset, train_args: TrainingArgs = None):
        pass
