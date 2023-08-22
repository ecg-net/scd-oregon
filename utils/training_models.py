import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import wandb
from sklearn import metrics as skl_metrics
from pathlib import Path
import pandas as pd


class TrainingMetric:
    def __init__(self, metric_func, metric_name, optimum=None):
        self.func = metric_func
        self.name = metric_name
        self.optimum = optimum

    def calc_metric(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        except ValueError as e:
            return np.nan

    def __call__(self, y_true, y_pred, split=None, step_type=None) -> dict:
        m = {
            f"{step_type}_{split}_{self.name}": self.calc_metric(
                y_true.flatten(), y_pred.flatten()
            )
        }

        return m


roc_auc_metric = TrainingMetric(skl_metrics.roc_auc_score, "roc_auc", optimum="max")


class BinaryClassificationModel(pl.LightningModule):
    def __init__(
        self,
        model,
        metrics=[roc_auc_metric],
        tracked_metric="roc_auc",
        early_stop_epochs=10,
        checkpoint_every_epoch=False,
        checkpoint_every_n_steps=None,
        index_labels=None,
        save_predictions_path=None,
        lr=0.01,
    ):
        super().__init__()
        self.epoch_preds = {"train": ([], []), "val": ([], [])}
        self.epoch_losses = {"train": [], "val": []}
        self.metrics = {}
        self.metric_funcs = {m.name: m for m in metrics}
        self.tracked_metric = f"epoch_val_{tracked_metric}"
        self.best_tracked_metric = None
        self.early_stop_epochs = early_stop_epochs
        self.checkpoint_every_epoch = checkpoint_every_epoch
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self.metrics["epochs_since_last_best"] = 0
        self.m = model
        self.training_steps = 0
        self.steps_since_checkpoint = 0
        self.labels = index_labels
        if self.labels is not None and isinstance(self.labels, str):
            self.labels = [self.labels]
        self.save_predictions_path = save_predictions_path
        self.lr = lr

        self.loss_func = nn.BCEWithLogitsLoss()

    def prepare_batch(self, batch):
        input_tensor, labels = super().prepare_batch(batch)
        if len(labels.shape) == 1:
            labels = labels[:, None]
        return input_tensor, labels

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

    def forward(self, x):
        return self.m(x)

    def prepare_batch(self, batch):
        filenames, input_tensor, labels = batch
        return input_tensor, labels

    def step(self, batch, step_type="train"):
        x, y_true = self.prepare_batch(batch)

        y_pred = self.forward(x)

        loss = self.loss_func(y_pred, y_true)
        if torch.isnan(loss):
            raise ValueError(loss)

        self.log_step(step_type, y_true, y_pred, loss)
        return loss

    def training_step(self, batch, i):
        return self.step(batch, "train")

    def validation_step(self, batch, i):
        return self.step(batch, "val")

    def predict_step(self, batch, *args):
        y_pred = self.forward(batch[1])
        return (batch[0], y_pred.cpu().numpy())

    def on_predict_epoch_end(self, results):
        for i, predict_results in enumerate(results):
            filename_df = pd.DataFrame(
                {"filename": np.concatenate([batch[0] for batch in predict_results])}
            )

            if self.labels is not None:
                columns = [f"{class_name}_preds" for class_name in self.labels]
            else:
                columns = ["preds"]

            outputs_df = pd.DataFrame(
                np.concatenate([batch[1] for batch in predict_results], axis=0),
                columns=columns,
            )
            prediction_df = pd.concat([filename_df, outputs_df], axis=1)

            dataloader = self.trainer.predict_dataloaders[i]
            manifest = dataloader.dataset.manifest
            prediction_df = prediction_df.merge(manifest, on="filename", how="outer")
            if wandb.run is not None:
                prediction_df.to_csv(
                    Path(wandb.run.dir).parent
                    / "data"
                    / f"dataloader_{i}_predictions.csv",
                    index=False,
                )
            if self.save_predictions_path is not None:
                prediction_df.to_csv(
                    self.save_predictions_path / f"dataloader_{i}_predictions.csv",
                    index=False,
                )
            if wandb.run is None and self.save_predictions_path is None:
                print(
                    "WandB is not active and self.save_predictions_path is None. Predictions will be saved to the directory this script is being run in."
                )
                prediction_df.to_csv(f"dataloader_{i}_predictions.csv", index=False)

    def log_step(self, step_type, labels, output_tensor, loss):
        self.epoch_preds[step_type][0].append(labels.detach().cpu().numpy())
        self.epoch_preds[step_type][1].append(output_tensor.detach().cpu().numpy())
        self.epoch_losses[step_type].append(loss.detach().item())
        if step_type == "train":
            self.training_steps += 1
            self.steps_since_checkpoint += 1
            if (
                self.checkpoint_every_n_steps is not None
                and self.steps_since_checkpoint > self.checkpoint_every_n_steps
            ):
                self.steps_since_checkpoint = 0
                self.checkpoint_weights(f"step_{self.training_steps}")

    def checkpoint_weights(self, name=""):
        if wandb.run is not None:
            weights_path = Path(wandb.run.dir).parent / "weights"
            if not weights_path.is_dir():
                weights_path.mkdir()
            torch.save(self.state_dict(), weights_path / f"model_{name}.pt")
        else:
            print("Did not checkpoint model. wandb not initialized.")

    def validation_epoch_end(self, preds):
        # Save weights
        self.metrics["epoch"] = self.current_epoch
        if self.checkpoint_every_epoch:
            self.checkpoint_weights(f"epoch_{self.current_epoch}")

        # Calculate metrics
        for m_type in ["train", "val"]:
            y_true, y_pred = self.epoch_preds[m_type]
            if len(y_true) == 0 or len(y_pred) == 0:
                continue
            y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)

            self.metrics[f"epoch_{m_type}_loss"] = np.mean(self.epoch_losses[m_type])
            for m in self.metric_funcs.values():
                self.metrics.update(
                    m(
                        y_true,
                        y_pred,
                        labels=self.labels,
                        split=m_type,
                        step_type="epoch",
                    )
                )

            # Reset predictions
            self.epoch_losses[m_type] = []
            self.epoch_preds[m_type] = ([], [])

        # Check if new best epoch
        if self.metrics is not None and self.tracked_metric is not None:
            if self.tracked_metric == "epoch_val_loss":
                metric_optimization = "min"
            else:
                metric_optimization = self.metric_funcs[
                    self.tracked_metric.replace("epoch_val_", "")
                ].optimum
            if (
                self.metrics[self.tracked_metric] is not None
                and (
                    self.best_tracked_metric is None
                    or (
                        metric_optimization == "max"
                        and self.metrics[self.tracked_metric] > self.best_tracked_metric
                    )
                    or (
                        metric_optimization == "min"
                        and self.metrics[self.tracked_metric] < self.best_tracked_metric
                    )
                )
                and self.current_epoch > 0
            ):
                print(
                    f"New best epoch! {self.tracked_metric}={self.metrics[self.tracked_metric]}, epoch={self.current_epoch}"
                )
                self.checkpoint_weights(f"best_{self.tracked_metric}")
                self.metrics["epochs_since_last_best"] = 0
                self.best_tracked_metric = self.metrics[self.tracked_metric]
            else:
                self.metrics["epochs_since_last_best"] += 1
            if self.metrics["epochs_since_last_best"] >= self.early_stop_epochs:
                raise KeyboardInterrupt("Early stopping condition met")

        # Log to w&b
        if wandb.run is not None:
            wandb.log(self.metrics)
