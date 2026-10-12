"""Train STFPM on MVTec AD with the current continual-baseline API."""
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-path", default="disk/checkpoints/stfpm.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    from moviad.datasets.dataset_arguments import DatasetArguments
    from moviad.datasets.mvtec import MVTecDataset
    from moviad.models.stfpm.stfpm import STFPM, STFPMTrainArgs
    from moviad.trainers.trainer import Trainer
    from moviad.utilities.configurations import Split
    from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
    from moviad.utilities.evaluation.evaluator import Evaluator
    from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers = ["layer1", "layer2", "layer3"]
    teacher = CustomFeatureExtractor("resnet18", layers, device, frozen=True)
    student = CustomFeatureExtractor("resnet18", layers, device, frozen=False)
    model = STFPM(teacher, student).to(device)
    dataset_args = DatasetArguments(
        dataset_path=args.dataset_path, img_size=(224, 224), gt_mask_size=(224, 224)
    )
    training = MVTecDataset(dataset_args, args.category, Split.TRAIN)
    testing = MVTecDataset(dataset_args, args.category, Split.TEST)
    metrics = [RocAuc(MetricLvl.IMAGE), RocAuc(MetricLvl.PIXEL)]
    trainer = Trainer(
        train_args=STFPMTrainArgs(batch_size=args.batch_size, epochs=args.epochs),
        model=model, train_dataset=training, eval_dataset=testing,
        metrics=metrics, device=device, save_path=None,
    )
    trainer.train()
    output = Path(args.save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    loader = torch.utils.data.DataLoader(testing, batch_size=args.batch_size)
    print(Evaluator.evaluate(model, loader, metrics, device))


if __name__ == "__main__":
    main()
