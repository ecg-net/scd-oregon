from pathlib import Path

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from utils.data import ECGDataset
from utils.models import EffNet
from utils.training_models import BinaryClassificationModel


def get_model(weights_path=None):
    backbone = EffNet(output_neurons=1)

    Classifier = BinaryClassificationModel(backbone, early_stop_epochs=100, lr=0.001)

    if weights_path is not None:
        print(Classifier.load_state_dict(torch.load(weights_path)))

    return Classifier


def train(
    data_path,
    manifest_path,
    batch_size,
    num_workers,
    gpus,
    max_epochs=100,
    weights_path=None,
):
    torch.cuda.empty_cache()

    train_ds = ECGDataset(
        split="train",
        data_path=data_path,
        labels="case",
        manifest_path=manifest_path,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=False,
    )

    val_ds = ECGDataset(
        split="val",
        data_path=data_path,
        labels="case",
        manifest_path=manifest_path,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )

    model = get_model(weights_path)

    trainer = Trainer(gpus=gpus, max_epochs=max_epochs, strategy="ddp")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    project = Path("/path/to/project")
    data_path = project / "train_normed_npy"
    manifest_path = "final_scd_manifest.csv"

    args = dict(
        max_epochs=1000,
        gpus=1,
        num_workers=18,
        batch_size=500,
        data_path=data_path,
        manifest_path=manifest_path,
        weights_path=None,
    )

    train(**args)
