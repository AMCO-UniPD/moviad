import wandb
import torch
from sklearn.cluster import MiniBatchKMeans

from tqdm import tqdm
import os

from moviad.models.patchcore.patchcore import PatchCore
from moviad.models.patchcore.kcenter_greedy import CoresetExtractor
from moviad.utilities.evaluation.evaluator import Evaluator
from moviad.trainers.trainer import Trainer, TrainerResult

class TrainerPatchCore(Trainer):

    """
    This class contains the code for training the CFA model

    Args:
        patchore_model (PatchCore): model to be trained
        train_dataloder (torch.utils.data.DataLoader): train dataloader
        test_dataloder (torch.utils.data.DataLoader): test dataloader
        device (str): device to be used for the training
    """

    def __init__(
        self,
        patchore_model: PatchCore,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloder: torch.utils.data.DataLoader,
        device: str,
        coreset_extractor: CoresetExtractor = None,
        save_path=None,
        logger=None,
    ):
        super().__init__(
            patchore_model, 
            train_dataloader, 
            test_dataloder, 
            device, 
            logger,
            save_path,
        )
        self.coreset_extractor = coreset_extractor

    def train(self):

        """
        This method trains the PatchCore model and evaluate it at the end of training
        """

        embeddings = []

        with torch.no_grad():

            if self.logger is not None:
                self.logger.watch(self.model)
            print("Embedding Extraction:")
            for batch in tqdm(iter(self.train_dataloader)):

                if isinstance(batch, tuple):
                    embedding = self.model(batch[0].to(self.device))
                else:
                    embedding = self.model(batch.to(self.device))

                #print(f"Embedding Shape: {embedding.shape}")


                embeddings.append(embedding)

            embeddings = torch.cat(embeddings, dim = 0)

            #print(f"Embeddings Shape: {embeddings.shape}")

            torch.cuda.empty_cache()

            # if self.model.apply_quantization:
            #     self.model.product_quantizer.fit(embeddings)
            #     embeddings = self.model.product_quantizer.encode(embeddings)


            #apply coreset reduction
            print("Coreset Extraction:")
            if self.coreset_extractor is None:
                self.coreset_extractor = CoresetExtractor(False, self.device, k=self.model.k)

            coreset = self.coreset_extractor.extract_coreset(embeddings)

            if self.model.apply_quantization:
                assert self.model.product_quantizer is not None, "Product Quantizer not initialized"

                self.model.product_quantizer.fit(coreset)
                coreset = self.model.product_quantizer.encode(coreset)

            self.model.memory_bank = coreset

            if self.save_path:
                self.model.save_model(save_path=self.save_path)

            gpu_device = torch.device("cuda:0")
            self.model.to(gpu_device)   
            self.evaluator.device = gpu_device

            metrics = self.evaluator.evaluate(self.model)

            if self.logger is not None:
                self.logger.log(
                    metrics
                )

            print("End training performances:")
            self.print_metrics(metrics)

            return TrainerResult(**metrics)







