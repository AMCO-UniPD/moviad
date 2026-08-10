import os
import gc
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.uniad.uniad import UniAD
from moviad.trainers.trainer_uniad import TrainerUniAD
from moviad.utilities.evaluator import Evaluator
from moviad.utilities.configurations import TaskType, Split


def main(args):

    device = torch.device(args.device)
    img_size = tuple(args.input_size)

    feature_extractor = CustomFeatureExtractor(
        args.backbone, args.ad_layers, device, True, False, None
    )

    train_dataset = MVTecDataset(
        TaskType.SEGMENTATION, args.dataset_path, args.category, "train", img_size=img_size
    )
    train_dataset.load_dataset()
    print(f"Train dataset size: {len(train_dataset)}")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = MVTecDataset(
        TaskType.SEGMENTATION, args.dataset_path, args.category,"test", img_size=img_size
    )
    test_dataset.load_dataset()
    print(f"Test dataset size: {len(test_dataset)}")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.train:
        print(f"\nTraining UniAD for category: {args.category}")

        model = UniAD(
            device=device,
            input_size=img_size,
            feature_extractor=feature_extractor,
        )
        model.to(device)
        model.train()

        trainer = TrainerUniAD(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            save_path=args.checkpoint_path,
        )

        result = trainer.train()

        print(f"\nResults for category: {args.category}")
        print(f"Image AUROC: {result.img_roc_auc:.4f}")
        print(f"Pixel AUROC: {result.pxl_roc_auc:.4f}")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    if args.eval:
        print(f"\nEvaluating UniAD for category: {args.category}")

        model = UniAD(
            device=device,
            input_size=img_size,
            feature_extractor=feature_extractor,
        )
        model.load_model(args.checkpoint_path)
        model.to(device)
        model.eval()

        evaluator = Evaluator(test_dataloader, device)
        metrics = evaluator.evaluate(model)

        print(f"\nResults for category: {args.category}")
        print(f"Image AUROC: {metrics['img_roc_auc']:.4f}")
        print(f"Pixel AUROC: {metrics['pxl_roc_auc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--ad_layers", type=str, nargs="+", default=["layer2"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--input_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--checkpoint_path", type=str, default=None)

    args = parser.parse_args()

    try:
        main(args)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("uniad.log", "a") as f:
            f.write("finished\t" + now_str + "\t" + str(args) + "\n")
    except Exception as e:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("uniad.log", "a") as f:
            f.write("** FAILED **\t" + now_str + "\t" + str(args) + "\n")
        raise e