from diffstack.modules.module import ModuleSequence, DataFormat
from trajdata.data_structures.batch import SceneBatch, AgentBatch
import pytorch_lightning as pl
import diffstack.utils.tensor_utils as TensorUtils
import diffstack.utils.geometry_utils as GeoUtils
from diffstack.utils.utils import removeprefix
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import wandb


class AVStack(pl.LightningModule):
    def __init__(self, modules: ModuleSequence, cfg, batch_size=None, **kwargs):
        super(AVStack, self).__init__()
        self.moduleseq = modules
        self.components = nn.ModuleDict()
        for k, v in self.moduleseq.components.items():
            self.components[k] = v
        self.cfg = cfg
        if "monitor_key" in kwargs:
            self.monitor_key = kwargs["monitor_key"]
        else:
            self.monitor_key = {"valLoss": "val/losses_predictor_prediction_loss"}

        self.batch_size = batch_size
        self.validation_step_outputs = []

    @property
    def input_format(self) -> DataFormat:
        self.moduleseq.input_format

    @property
    def output_format(self) -> DataFormat:
        self.moduleseq.output_format

    @property
    def checkpoint_monitor_keys(self):
        return self.monitor_key

    def forward(self, inputs, **kwargs):
        return self.moduleseq(inputs, **kwargs)

    def infer_step(self, inputs, **kwargs):
        with torch.no_grad():
            return TensorUtils.detach(self.moduleseq.infer_step(inputs, **kwargs))

    def _compute_losses(self, pout, inputs):
        loss_by_component = dict()
        for comp_name, component in self.moduleseq.components.items():
            comp_out = {
                removeprefix(k, comp_name + "."): v
                for k, v in pout.items()
                if k.startswith(comp_name + ".")
            }
            loss_by_component[comp_name] = component.compute_losses(comp_out, inputs)

        return loss_by_component

    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            total_loss(torch.Tensor): total weighted loss
        """

        pout = self(batch, batch_idx=batch_idx)
        losses = self._compute_losses(pout, batch)

        total_loss = 0.0
        for comp_name in self.moduleseq.components:
            comp_loss = losses[comp_name]
            for lk, l in comp_loss.items():
                loss = l * self.cfg.stack[comp_name].loss_weights[lk]
                self.log("train/losses_" + comp_name + "_" + lk, loss, sync_dist=True)
                total_loss += loss
        if batch_idx % 100 == 0:
            metrics = self._compute_metrics(pout, batch)
            for mk, m in metrics.items():
                if isinstance(m, np.ndarray) or isinstance(m, torch.Tensor):
                    m = m.mean()
                self.log("train/metrics_" + mk, m, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            pout = TensorUtils.detach(
                self.moduleseq.validate_step(batch, batch_idx=batch_idx)
            )
        losses = self._compute_losses(pout, batch)

        if self.logger is not None and batch_idx == 1:
            if "predictor" in self.components and hasattr(
                self.components["predictor"], "log_pred_image"
            ):
                self.components["predictor"].log_pred_image(
                    batch, pout, batch_idx, self.logger
                )
            else:
                # default
                self.log_pred_image(batch, pout, batch_idx, self.logger)
            print("image logged")

        metrics = self._compute_metrics(pout, batch)
        pred = {"losses": losses, "metrics": metrics}
        self.validation_step_outputs.append(pred)
        return pred

    def test_step(self, batch, batch_idx):
        # if batch_idx<380 or batch_idx>381:
        #     return {}
        # if "log_image_frequency" in self.cfg.eval and self.cfg.eval.log_image_frequency is not None:
        #     if batch_idx%self.cfg.eval.log_image_frequency!=0:
        #         return {}
        with torch.no_grad():
            pout = TensorUtils.detach(
                self.moduleseq.validate_step(batch, batch_idx=batch_idx)
            )
            losses = self._compute_losses(pout, batch)
            metrics = self._compute_metrics(pout, batch)
            pred = {"losses": losses, "metrics": metrics}
        flattened_pred = TensorUtils.flatten_dict(pred)
        for k, v in flattened_pred.items():
            if isinstance(v, torch.Tensor):
                flattened_pred[k] = (
                    v.cpu().numpy().item()
                    if v.numel() == 1
                    else v.cpu().numpy().mean().item()
                )
            elif isinstance(v, np.ndarray):
                flattened_pred[k] = v.item() if v.size == 1 else v.mean().item()

        self.log_dict(flattened_pred)

        if (
            "log_image_frequency" in self.cfg.eval
            and self.cfg.eval.log_image_frequency is not None
        ):
            if batch_idx % self.cfg.eval.log_image_frequency == 0:
                for comp_name, component in self.moduleseq.components.items():
                    if hasattr(component, "log_pred_image"):
                        component.log_pred_image(
                            batch,
                            pout,
                            batch_idx,
                            self.cfg.eval.results_dir,
                            log_all_image=self.cfg.eval.log_all_image,
                            savegif=self.cfg.eval.get("savegif", True),
                        )

        return flattened_pred

    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs
        for comp_name in self.moduleseq.components:
            for k in outputs[0]["losses"][comp_name]:
                m = torch.stack([o["losses"][comp_name][k] for o in outputs]).mean()
                self.log("val/losses_" + comp_name + "_" + k, m, sync_dist=True)

        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m, sync_dist=True)
        self.validation_step_outputs = []

    def configure_optimizers(self):
        pass

    def _compute_metrics(self, pred_batch, data_batch):
        pass

    def log_pred_image(
        self,
        batch,
        pred,
        batch_idx,
        logger,
        **kwargs,
    ):
        if "pred" in pred:
            pred = pred["pred"]
        if "image" in pred:
            N = pred["image"].shape[0]
            h = int(np.ceil(np.sqrt(N)))
            w = int(np.ceil(N / h))
            fig, ax = plt.subplots(h, w, figsize=(20 * h, 20 * w))
            canvas = FigureCanvas(fig)
            image = pred["image"].detach().cpu().numpy().transpose(0, 2, 3, 1)
            for i in range(N):
                if h > 1 and w > 1:
                    hi = int(i / w)
                    wi = i - w * hi
                    ax[hi, wi].imshow(image[i])

                elif h == 1 and w == 1:
                    ax.imshow(image[i])
                else:
                    ax[i].imshow(image[i])
            canvas.draw()  # draw the canvas, cache the renderer

            image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            logger.experiment.log(
                {"val_pred_image": [wandb.Image(image, caption="val_pred_image")]}
            )
            del fig, ax, image
        else:
            if "data_batch" in pred:
                batch = pred["data_batch"]
            else:
                if "scene_batch" in batch:
                    batch = batch["scene_batch"]
                elif "agent_batch" in batch:
                    batch = batch["agent_batch"]
                if isinstance(batch, SceneBatch) or isinstance(batch, AgentBatch):
                    batch = batch.__dict__

            if "trajectories" not in pred or "image" not in batch:
                return

            keys = ["trajectories", "agent_avail", "cond_traj"]
            pred = {k: v for k, v in pred.items() if k in keys}
            pred = TensorUtils.to_numpy(pred)

            if batch["image"].ndim == 5:
                map = batch["image"][0, 0, -3:].cpu().numpy().transpose(1, 2, 0)
            else:
                map = batch["image"][0, -3:].cpu().numpy().transpose(1, 2, 0)
            traj = pred["trajectories"][..., :2]

            avail = pred["agent_avail"].astype(float)
            avail1 = avail.copy()
            avail1[avail == 0] = np.nan
            raster_from_agent = batch["raster_from_agent"].cpu().numpy()
            if raster_from_agent.ndim == 2:
                raster_from_agent = raster_from_agent[np.newaxis, :]
            if traj.ndim == 6:
                bs, Ne, numMode, Na, T = traj.shape[:5]
                traj = traj * avail1[:, None, None, :, None, None]
            else:
                bs, numMode, Na, T = traj.shape[:4]
                traj = traj * avail1[:, None, :, None, None]
                Ne = 1
            raster_traj = GeoUtils.batch_nd_transform_points_np(
                traj.reshape(bs, -1, 2), raster_from_agent
            ).reshape(bs, Ne, -1, 2)

            cond_traj = pred["cond_traj"] if "cond_traj" in pred else None

            if cond_traj is None:
                fig, ax = plt.subplots(figsize=(20, 20))
                canvas = FigureCanvas(fig)
                ax.imshow(map)
                ax.scatter(
                    raster_traj[0, 0, ..., 0],
                    raster_traj[0, 0, ..., 1],
                    color="c",
                    s=2,
                    marker="D",
                )

            else:
                raster_cond_traj = GeoUtils.batch_nd_transform_points_np(
                    cond_traj[..., :2].reshape(bs, -1, 2), raster_from_agent
                ).reshape(bs, Ne, -1, 2)
                h = int(np.ceil(np.sqrt(Ne)))
                w = int(np.ceil(Ne / h))
                fig, ax = plt.subplots(h, w, figsize=(20 * h, 20 * w))
                canvas = FigureCanvas(fig)
                for i in range(Ne):
                    if h > 1 and w > 1:
                        hi = int(i / w)
                        wi = i - w * hi
                        ax[hi, wi].imshow(map)
                        ax[hi, wi].scatter(
                            raster_traj[0, i, ..., 0],
                            raster_traj[0, i, ..., 1],
                            color="c",
                            s=1,
                            marker="D",
                        )
                        ax[hi, wi].scatter(
                            raster_cond_traj[0, i, :, 0],
                            raster_cond_traj[0, i, :, 1],
                            color="m",
                            s=1,
                            marker="D",
                        )
                    elif h == 1 and w == 1:
                        ax.imshow(map)
                        ax.scatter(
                            raster_traj[0, i, ..., 0],
                            raster_traj[0, i, ..., 1],
                            color="c",
                            s=1,
                            marker="D",
                        )
                        ax.scatter(
                            raster_cond_traj[0, i, :, 0],
                            raster_cond_traj[0, i, :, 1],
                            color="m",
                            s=1,
                            marker="D",
                        )
                    else:
                        ax[i].imshow(map)
                        ax[i].scatter(
                            raster_traj[0, i, ..., 0],
                            raster_traj[0, i, ..., 1],
                            color="c",
                            s=1,
                            marker="D",
                        )
                        ax[i].scatter(
                            raster_cond_traj[0, i, :, 0],
                            raster_cond_traj[0, i, :, 1],
                            color="m",
                            s=1,
                            marker="D",
                        )
            canvas.draw()  # draw the canvas, cache the renderer

            image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            logger.experiment.log(
                {"val_pred_image": [wandb.Image(image, caption="val_pred_image")]}
            )
            del fig, ax, image

    def set_eval(self):
        self.moduleseq.set_eval()

    def set_train(self):
        self.moduleseq.set_train()

    def reset(self):
        self.moduleseq.reset()
