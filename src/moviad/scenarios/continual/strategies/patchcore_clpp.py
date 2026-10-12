from moviad.datasets.vad_dataset import VADDataset
from moviad.scenarios.continual.continual_model import ContinualModel
from moviad.models.patchcore.patchcore import PatchCore
from moviad.models.training_args import TrainingArgs
from moviad.utilities.get_sizes import count_params, params_to_mb

import torch
from tqdm import tqdm


class PatchCoreCLPP(ContinualModel):

    def __init__(self, patchcore_model: PatchCore):
        super().__init__(patchcore_model)

        # change the patchcore memory bank with a set of memory banks, one per task
        self.vad_model.memory_bank = {}
        self.task_prototypes = []
        self.n_samples_per_task = self.vad_model.memory_bank_size

    def _rebalance_memory_bank(self):
        n_tasks = len(self.vad_model.memory_bank) + 1
        self.n_samples_per_task = self.vad_model.memory_bank_size // n_tasks
        self.vad_model.coreset_extractor.k = self.n_samples_per_task

        for task_id in self.vad_model.memory_bank:
            print("Rebalancing Memory Bank for Task", task_id, "with new sample limit:", self.n_samples_per_task)
            self.vad_model.memory_bank[task_id] = self.vad_model.memory_bank[task_id][:self.n_samples_per_task]

    @staticmethod
    def get_prototype(feature_extractor, batch):
        last_features = feature_extractor(batch)[-1]
        return last_features.mean(dim=[2,3])


    def _update_prototypes(self, task_index: int, train_dataset: VADDataset, train_args: TrainingArgs = None):
        print("Adding Prototype for Task", task_index)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_args.batch_size, shuffle=True)

        task_prototypes = []

        for batch in tqdm(train_dataloader):

            batch=batch.to(self.vad_model.device)
            with torch.no_grad():
                features = PatchCoreCLPP.get_prototype(self.vad_model.feature_extractor, batch)
            task_prototypes.append(features.mean(dim=0))

        self.task_prototypes.append(torch.stack(task_prototypes).mean(dim=0))


    def start_task(self, task_index: int, train_dataset: VADDataset, train_args: TrainingArgs = None):
        self._rebalance_memory_bank()
        self._update_prototypes(task_index, train_dataset, train_args)


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

            #apply coreset reduction
            print("Coreset Extraction:")
            coreset = self.vad_model.coreset_extractor.extract_coreset(embeddings)

            self.vad_model.memory_bank[task_index] = coreset


    def end_task(self, task_index:int, train_dataset: VADDataset, train_args: TrainingArgs = None):
        pass

    def forward(self, batch: torch.Tensor):
        # 1. Compute distances to task prototypes
        batch_prototypes = PatchCoreCLPP.get_prototype(self.vad_model.feature_extractor, batch)

        # Safely convert the list of prototypes to a tensor for this forward pass only
        if isinstance(self.task_prototypes, list):
            task_protos_tensor = torch.stack(self.task_prototypes).to(self.vad_model.device)
        else:
            task_protos_tensor = self.task_prototypes

        # Calculate norms and distances
        proto_norms = (task_protos_tensor ** 2).sum(dim=1).unsqueeze(0)  # [1, N]
        vec_norms = (batch_prototypes ** 2).sum(dim=1).unsqueeze(1)      # [B, 1]

        # Now the shapes will correctly be: [B, 1] + [1, N] - 2 * ([B, 160] @ [160, N])
        dists = vec_norms + proto_norms - 2 * batch_prototypes @ task_protos_tensor.T  # [B, N]

        indices = torch.argmin(dists, dim=1)  # [B]
        self.loaded_adapters_ids = indices.cpu().numpy()

        # 2. Group the batch by assigned task to process efficiently
        unique_tasks = torch.unique(indices)

        batch_scores = torch.zeros(batch.size(0), device=batch.device)
        batch_maps = [None] * batch.size(0)

        memory_banks_dict = self.vad_model.memory_bank

        for task_id in unique_tasks:
            task_id_val = task_id.item()

            mask = indices == task_id_val
            original_indices = torch.where(mask)[0]

            sub_batch = batch[mask]

            # ---> CRITICAL FIX: You must assign the correct memory bank for this sub-batch! <---
            self.vad_model.memory_bank = memory_banks_dict[task_id_val]

            sub_maps, sub_scores = self.vad_model(sub_batch)

            batch_scores[mask] = sub_scores

            for i, orig_idx in enumerate(original_indices):
                batch_maps[orig_idx.item()] = sub_maps[i]

        # Restore the memory bank dictionary
        self.vad_model.memory_bank = memory_banks_dict

        if len(batch_maps) > 0 and isinstance(batch_maps[0], torch.Tensor):
            batch_maps = torch.stack(batch_maps)

        return batch_maps, batch_scores

    def get_model_size(self):
        feature_extractor_size = self.vad_model.feature_extractor.get_size()

        memory_bank_size = 0
        for task_id in self.vad_model.memory_bank:
            memory_bank_size += params_to_mb(count_params(self.vad_model.memory_bank[task_id]))

        return {
            "feature_extractor": feature_extractor_size,
            "memory_bank": memory_bank_size,
            "total": feature_extractor_size + memory_bank_size
        }
