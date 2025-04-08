import gc
import torch
from dataclasses import dataclass
from moviad.common.args import Args
from moviad.datasets.common import IadDataset
from moviad.entrypoints.common import load_datasets
from moviad.models.rd4ad import RD4AD
from moviad.trainers.trainer_rd4ad import RD4AD_Trainer


@dataclass
class RD4ADArgs(Args):
    train_dataset: IadDataset = None
    test_dataset: IadDataset = None
    category: str = None
    backbone: str = None
    ad_layers: list = None
    img_input_size: tuple = (224, 224)
    batch_size: int = 2
    device: torch.device = None


def train_rd4ad(args: RD4ADArgs, logger=None) -> None:

    train_dataset, test_dataset = load_datasets(args.dataset_config, args.dataset_type, args.category, image_size=args.img_input_size)


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   drop_last=True)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                                  drop_last=True)

    # define the model
    model = RD4AD(args.device,args.img_input_size)
    model.to(args.device)
    trainer = RD4AD_Trainer(model, train_dataloader, test_dataloader, args.device, logger=logger)
    trainer.train()

    # save the model
    if args.save_path:
        torch.save(model.state_dict(), args.save_path)

    # force garbage collector in case
    del patchcore
    del train_dataloader
    del test_dataloader
    torch.cuda.empty_cache()
    gc.collect()
