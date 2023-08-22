import glob
import shutil
from collections import namedtuple
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from utils.data import ECGDataset
from utils.models import EffNet
from utils.training_models import BinaryClassificationModel

project = Path("/path/to/project")
Run = namedtuple("Run", ["name", "data_path", "manifest_path", "weights", "split"])

weights = torch.load("/path/to/weights")

runs = [
    Run(
        "sudden_cardiac_death_final",
        project / "train_normed_npy",
        "sudden_cardiac_death_final.csv",
        weights,
        None,
    )
]

for run in runs:
    test_ds = ECGDataset(
        split=run.split,
        data_path=run.data_path,
        manifest_path=run.manifest_path,
        labels="case",
    )

    test_dl = DataLoader(
        test_ds, num_workers=18, batch_size=500, drop_last=False, shuffle=False
    )

    backbone = EffNet(output_neurons=1)
    classifier = BinaryClassificationModel(backbone)

    print(classifier.load_state_dict(run.weights))

    trainer = Trainer(gpus=1)

    trainer.predict(classifier, dataloaders=test_dl)

    for results_csv in glob.glob("dataloader_*_predictions.csv"):
        results_csv = Path(results_csv)
        shutil.move(results_csv, f"results/{run.name}_predictions.csv")
