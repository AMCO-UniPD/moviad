import torch
from tqdm import tqdm

from moviad.models.uniad.uniad import UniAD
from moviad.utilities.evaluator import Evaluator
from moviad.trainers.trainer import Trainer, TrainerResult


class TrainerUniAD(Trainer):
    """Trainer for the UniAD model.
    Follows the same pattern as other MoViAD trainers.
    """
    def __init__(
        self,
        model: UniAD,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        epochs: int = 50,
        lr: float = 1e-4,
        save_path: str = None,
        logger=None,
    ):
        super().__init__(model, train_dataloader, test_dataloader, device, logger, save_path)
        self.epochs = epochs
        self.lr = lr

    def train(self):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.uniad.parameters(), lr=self.lr)
        criterion = torch.nn.MSELoss()

        print(f"Training UniAD for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            total_loss = 0.0

            for batch in tqdm(iter(self.train_dataloader), desc=f"Epoch [{epoch+1}/{self.epochs}]"):
                images = batch[0].to(self.device) if isinstance(batch, (tuple, list)) else batch.to(self.device)

                optimizer.zero_grad()
                output = self.model(images)
                loss = criterion(output["feature_rec"], output["feature_align"])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_dataloader)
            print(f"Epoch [{epoch+1}/{self.epochs}] Loss: {avg_loss:.4f}")

        if self.save_path:
            self.model.save_model(self.save_path)

        print("Training complete. Running evaluation...")
        self.model.eval()
        metrics = self.evaluator.evaluate(self.model)

        if self.logger is not None:
            self.logger.log(metrics)

        print("Evaluation results:")
        self.print_metrics(metrics)

        return TrainerResult(**metrics)