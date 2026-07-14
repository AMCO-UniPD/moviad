def test_model_create_train():
    from moviad.models.anomalyclip.anomalyclip import AnomalyCLIP, AnomalyCLIPArgs
    from moviad.models.training_args import TrainingArgs
    from moviad.trainers.trainer import Trainer
    from moviad.trainers.trainer_anomalyclip import AnomalyCLIPTrainer
    from moviad.datasets.cps_ad2d.cpsad2d_dataset import CPSAD2DDataset
    from moviad.datasets.dataset_arguments import DatasetArguments
    from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc
    from torch.utils.data import DataLoader, Subset
    import torch
    import wandb
    from tqdm import tqdm
    import numpy as np
    from moviad.utilities.evaluation.metrics import MetricLvl
    from torchvision import transforms
    from moviad.backbones.clip import constants
    from moviad.utilities.evaluation.evaluator import Evaluator

   

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 336

    transform_list = [
      transforms.ToTensor(),
      transforms.Resize((img_size, img_size)),
      transforms.Normalize(mean=constants.OPENAI_DATASET_MEAN, std=constants.OPENAI_DATASET_STD)
      ]

    wandb.init(project="anomaly_clip_test", entity = "test_cpsas2d", name="anomaly_clip", mode="disabled")


    args = DatasetArguments(
        dataset_path = "/home/nicola_berti/big_disk/Datasets/CPS-AD2D",
        img_size = (img_size, img_size),
        gt_mask_size = (img_size, img_size),
        image_transform_list = transform_list
    )

    # IMPORTANT : the actual training dataset cannot be composed only by normal samples.
    # This is why for testing the training I am using th test dataset (default should be 15 epoch).
    train_normal_dataset = CPSAD2DDataset(args, split="train")
    test_dataset = CPSAD2DDataset(args, split="test")

    print(f"Train dataset size: {len(train_normal_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    #TEST TRAINING
    model = AnomalyCLIP(
        device=device,
        features_list=[6, 12, 18, 24],
        feature_map_layer=[0, 1, 2, 3],
        dpam_layer=20,
        n_ctx=12,
        depth=9,
        t_n_ctx=4,
        pretrained_model="ViT-L/14@336px",
        checkpoint= "visa", # or "mvtec" or None
        sigma = 4,
    )
    model.to(device)

    training_args = AnomalyCLIPArgs(batch_size=32, epochs=1, evaluation_epoch_interval=2)

    trainer = AnomalyCLIPTrainer(
        training_args,
        model,
        test_dataset,
        test_dataset,
        metrics=[
            RocAuc(MetricLvl.IMAGE),
            RocAuc(MetricLvl.PIXEL),
            AvgPrec(MetricLvl.IMAGE),
            AvgPrec(MetricLvl.PIXEL),
            F1(MetricLvl.IMAGE),
            F1(MetricLvl.PIXEL),
            ProAuc(MetricLvl.PIXEL),
        ],
        device=device,
        logger=wandb,
        save_path="/home/nicola_berti/big_disk/AnomalyCLIP_checkpoint/save_test.pth",
        saving_criteria=None,
    )

    trainer.train()


    #TEST ZERO_SHOT/FEW_SHOT

    k_values = [0, 1, 2, 4, 8]

    for k in k_values:

        model = AnomalyCLIP(
            device=device,
            features_list=[6, 12, 18, 24],
            feature_map_layer=[0, 1, 2, 3],
            dpam_layer=20,
            n_ctx=12,
            depth=9,
            t_n_ctx=4,
            pretrained_model="ViT-L/14@336px",
            checkpoint= "visa", # or "mvtec" or None
            sigma = 4,
            alpha = 0.5
        )
        model.to(device)
        model.eval()

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2
        ) 

        if k > 0:

            few_shot_indices = list(range(min(k, len(train_normal_dataset))))
            #few_shot_indices = centroids_indices[k]
            few_shot_subset = Subset(train_normal_dataset, few_shot_indices)
            few_shot_dataloader = DataLoader(few_shot_subset, batch_size=1, shuffle=False, num_workers=2)

            model.build_memory_bank(few_shot_dataloader)

        results = Evaluator.evaluate(model, test_dataloader, metrics=[
                RocAuc(MetricLvl.IMAGE),
                RocAuc(MetricLvl.PIXEL),
                AvgPrec(MetricLvl.IMAGE),
                AvgPrec(MetricLvl.PIXEL),
                F1(MetricLvl.IMAGE),
                F1(MetricLvl.PIXEL),
                ProAuc(MetricLvl.PIXEL),
            ], device=device )
        
        # --- PRINT ---
        print("\n" + "="*50)
        print(f" REPORT FINALE - Valore k: {k} ".center(50, "="))
        print("="*50)
        for metric_name, value in results.items():
            print(f"  {metric_name:<25} : {value:.4f}")
        print("="*50 + "\n")

        # --- SALVATAGGIO WANDB ---
        if wandb:
            log_data = {f"test/{metric_name}": value for metric_name, value in results.items()}
            log_data["k_value"] = k
            wandb.log(log_data, step=k)


def setup_seed(seed):
    import torch
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    setup_seed(42)
    test_model_create_train()
    


                                                     