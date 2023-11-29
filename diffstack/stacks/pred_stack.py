from diffstack.modules.module import Module, ModuleSequence, DataFormat
from trajdata import SceneBatch
import pytorch_lightning as pl
import diffstack.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from diffstack.stacks.base import AVStack


class PredStack(AVStack):
    def __init__(self, modules: ModuleSequence, cfg, batch_size=None, **kwargs):
        super().__init__(
            modules, cfg, batch_size=cfg.train.training.batch_size, **kwargs
        )

    def configure_optimizers(self):
        optim_params = self.cfg.stack["predictor"].optim_params["policy"]
        return optim.Adam(
            params=self.parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["regularization"]["L2"],
        )

    def _compute_metrics(self, pred_batch, data_batch):
        metrics_dict = dict()
        if hasattr(self.components["predictor"], "compute_metrics"):
            metrics_dict.update(
                self.components["predictor"].compute_metrics(pred_batch, data_batch)
            )
        if (
            "trajectories" in pred_batch
            and pred_batch["trajectories"].ndim == 5
            and pred_batch["trajectories"].size(1) > 0
        ):
            metrics_dict["mode_diversity"] = (
                (pred_batch["trajectories"][:, 0] - pred_batch["trajectories"][:, 1])
                .norm()
                .detach()
                .cpu()
            )

        return metrics_dict
