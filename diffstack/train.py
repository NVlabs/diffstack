import sys

import torch
import numpy as np
import os
import time
import json
import pathlib
import wandb

from collections import defaultdict
from tqdm import tqdm, trange
from torch import nn, optim
from typing import Dict
from random import random

# Dataset related
from diffstack.data.trajdata_interface import prepare_avdata
from diffstack.data.cached_nusc_as_trajdata import prepare_cache_to_avdata
from diffstack.utils.model_registrar import ModelRegistrar

# Model related
from diffstack.modules.diffstack import DiffStack
from diffstack.argument_parser import args, get_hyperparams, print_hyperparams_summary
# from diffstack.closed_loop_eval import simulate_scenarios_in_scene

from diffstack.utils.utils import initialize_torch_distributed, prepeare_torch_env, set_all_seeds, all_gather
from torch.nn.parallel import DistributedDataParallel as DDP

import matplotlib.pyplot as plt

# Visualization
from diffstack.utils import visualization as plan_vis

from trajdata import AgentBatch, AgentType


def train(rank, args):
    hyperparams = get_hyperparams(args)
    # del args

    prepeare_torch_env(rank, hyperparams)
    log_writer, model_dir = prepare_logging(rank, hyperparams)

    if rank == 0:
        print_hyperparams_summary(hyperparams)

    device = hyperparams["device"]

    #################################
    #      PREPARE MODEL AND DATA   #
    #################################

    # Create model first. We need this before caching training data with gt plan.
    model_registrar = ModelRegistrar(model_dir, device)
    if hyperparams["load"]:
        # Directly adding to pythonpath for legacy reasons.
        from diffstack.modules.predictors import trajectron_utils
        sys.path.append(trajectron_utils.__path__[0]) 
        model_registrar.load_model_from_file(hyperparams["load"], except_contains=["planner_cost"])

    diffstack = DiffStack(model_registrar, hyperparams, log_writer, device)
    print(f'Rank {rank}: Created Training Model.')

    if hyperparams["data_source"] == "trajdata":
        train_dataloader, train_sampler, train_dataset, eval_dataloader, eval_sampler, eval_dataset, input_wrapper = prepare_avdata(rank, hyperparams, scene_centric=False)
    elif hyperparams["data_source"] == "trajdata-scene":
        train_dataloader, train_sampler, train_dataset, eval_dataloader, eval_sampler, eval_dataset, input_wrapper = prepare_avdata(rank, hyperparams, scene_centric=True)
    elif hyperparams["data_source"] == "cache":
        # Load original cached data and convert it to trajdata
        train_dataloader, train_sampler, train_dataset, eval_dataloader, eval_sampler, eval_dataset, input_wrapper = prepare_cache_to_avdata(rank, hyperparams, args, diffstack)
    else:
        raise ValueError(f"Unknown data_source {hyperparams['data_source']}")

    if torch.cuda.is_available() and device != 'cpu':
        diffstack = DDP(diffstack,
                        device_ids=[rank],
                        output_device=rank,
                        find_unused_parameters=True)
        diffstack_module = diffstack.module
        # pkarkus: DDP moves tensors to GPU (except for dilled dicts when using more than 1 worker)
        # using this function we can replicate the same for eval.
        # input_wrapper = lambda inputs, **kwargs: trajectron.to_kwargs(inputs, kwargs, trajectron.device_ids[0])
    else:
        diffstack_module = diffstack
        # input_wrapper = lambda inputs, **kwargs: (inputs, kwargs)

    # Initialize optimizer
    lr_scheduler = None
    step_scheduler = None
    plan_cost_lr = hyperparams['cost_grad_scaler'] * hyperparams['learning_rate']
    optimizer = optim.Adam([{'params': model_registrar.get_all_but_name_match('map_encoder').parameters()},
                            {'params': model_registrar.get_name_match('map_encoder').parameters(),
                             'lr': hyperparams['map_enc_learning_rate']},
                            {'params': model_registrar.get_name_match('planner_cost').parameters(), 
                             'lr': plan_cost_lr}
                            ], 
                           lr=hyperparams['learning_rate'])
    # Set Learning Rate
    if hyperparams['learning_rate_style'] == 'const':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    elif hyperparams['learning_rate_style'] == 'exp':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                        gamma=hyperparams['learning_decay_rate'])

    if hyperparams['lr_step'] != 0:
        step_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=hyperparams['lr_step'], gamma=0.1)


    ##################################
    #          VALIDATE FUNCTIONS        #
    ##################################
    def log_batch_errors(batch_errors, eval_metrics,
                        log_writer, namespace, epoch, curr_iter, 
                        bar_plot=[], box_plot=[]):
        for node_type in batch_errors.keys():
            if eval_metrics is None:
                # Log all metrics if eval_metrics is not provided
                eval_metrics = batch_errors[node_type].keys()
            for metric in eval_metrics:
                metric_batch_error: np.ndarray = np.concatenate(batch_errors[node_type][metric])

                if len(metric_batch_error) > 0 and log_writer is not None:
                    log_writer.log({
                        # f"{node_type.name}/{namespace}/{metric}_hist": wandb.Histogram(metric_batch_error),
                        # f"{node_type.name}/{namespace}/{metric}_min": np.min(metric_batch_error),
                        f"{node_type.name}/{namespace}/{metric}_mean": np.mean(metric_batch_error),
                        # f"{node_type.name}/{namespace}/{metric}_median": np.median(metric_batch_error),
                        # f"{node_type.name}/{namespace}/{metric}_max": np.max(metric_batch_error),
                        f"{node_type.name}/epoch": epoch,  # TODO(pkarkus) this is only for compatibility, remove it
                    }, step=curr_iter, commit=False)

        if log_writer is not None:
            log_writer.log({"epoch": epoch, "global_step": curr_iter}, step=curr_iter, commit=True)

    def run_validation(epoch, curr_iter):
        with torch.no_grad():
            # Calculate evaluation loss
            epoch_metrics: Dict[str, list] = defaultdict(list)
            
            # Compute metrics over validation set
            batch: AgentBatch
            for batch in tqdm(eval_dataloader, ncols=80, unit_scale=hyperparams["world_size"], 
                                disable=(rank > 0), desc=f'Epoch {epoch} Eval'):

                with torch.no_grad():
                    outputs = diffstack_module.validate(input_wrapper(batch))
                metrics: Dict[str, torch.Tensor] = outputs["metrics"]

                for k, v in metrics.items():
                    epoch_metrics[k].append(v.cpu().numpy())

            # Gather results from GPUs
            if hyperparams["world_size"] > 1:
                gathered_values = all_gather(epoch_metrics)
                
                if rank == 0:
                    epoch_metrics = defaultdict(list)
                    for partial_epoch_metrics in gathered_values:
                        for k, v in partial_epoch_metrics.items():
                            epoch_metrics[k].extend(v)

            # Log
            if rank == 0:
                log_batch_errors({AgentType.UNKNOWN: epoch_metrics},
                                            None,
                                            log_writer,
                                            'eval',
                                            epoch,
                                            curr_iter)
                eval_loss_sorted = {k: np.sort(np.concatenate(vals)) for k, vals in epoch_metrics.items()}
                print (f"Eval epoch {epoch}:")
                for topn in [100]:  # , 20, 10]:
                    print (f"Top {topn}%:")
                    for k, vals in eval_loss_sorted.items():
                        start_i = (100-topn)*len(vals)//100
                        topn_vals = vals[start_i:]
                        print (f"  {k} {topn}: {topn_vals.mean():.4f}")
                    
                    pass


    def run_closed_loop_eval(epoch, curr_iter, all_scenarios_in_scene=False):
        if hyperparams["cl_trajlen"] <= 0:
            print ("No closed loop evaluation.")
            return

        raise NotImplementedError("Need to access scenes in environment dataset and run custom preprocessing")

        # Run closed loop replanning at different frequencies
        replan_every_ns = [1]  #, 4, 5, 6]

        with torch.no_grad():
            # Calculate evaluation loss
            for node_type, data_loader in eval_data_loader.items():
                if rank == 0:
                    print(f"Starting closed loop evaluation @ epoch {epoch} for node type: {node_type}")

                # TODO support more workers
                env = eval_dataset.env
                scenes_for_worker = env.scenes
                if rank == 0:
                    for scene in tqdm(scenes_for_worker, disable=(rank > 0), desc=f'Scene '):
                        eval_loss = simulate_scenarios_in_scene(diffstack_module, nusc_maps, env, scene, node_type, hyperparams, replan_every_ns=replan_every_ns, all_scenarios_in_scene=False)
                else:
                    eval_loss = defaultdict(list)


                if torch.distributed.get_world_size() > 1:
                    gathered_values = all_gather(eval_perf)
                    if rank == 0:
                        eval_perf = []
                        for eval_dicts in gathered_values:
                            eval_perf.extend(eval_dicts)             
               
                if rank == 0:
                    log_batch_errors(eval_loss,
                                                None,
                                                log_writer,
                                                'eval',
                                                epoch,
                                                curr_iter) 
                    eval_loss_sorted = {k: np.sort(np.concatenate(vals)) for k, vals in eval_loss.items()}
                    print (f"Eval epoch {epoch}:")
                    for topn in [100]:
                        print (f"Top {topn}%%:")
                        for k, vals in eval_loss_sorted.items():
                            start_i = (100-topn)*len(vals)//100
                            topn_vals = vals[start_i:]
                            print (f"  {k} {topn}: {topn_vals.mean():.6f}")
                        
                        pass

    #################################
    #          VISUALIZATION        #
    #################################

    def run_visualization(epoch, curr_iter, dataset_name, dataset = None, dataloader = None, num_plots=10):
        with torch.no_grad():
            if dataset is not None:
                batch_idxs = random.sample(range(len(dataset)), num_plots)
                batch: AgentBatch = dataset.get_collate_fn(pad_format="right")(
                    [dataset[i] for i in batch_idxs]
                )
            elif dataloader is not None:
                for batch in dataloader:
                    break
            else:
                raise ValueError("Need to specifiy dataset or data_loader")

            outputs = diffstack_module.validate(input_wrapper(batch))
            plan_xu = outputs['plan.plan_xu'].cpu().numpy()

            # Only keep elements in batch that are valid for planning
            batch = batch.filter_batch(outputs['plan.valid'])

            # TODO this is not fixed, wrongly indexing plan output
            images = list()
            for batch_idx in trange(min(batch.agent_fut.shape[0], num_plots), desc="Visualizing Random Predictions"):
                plan_x = plan_xu[:, batch_idx, :4]

                fig, ax = plt.subplots()
                if 'plan.fan.candidate_xu' in outputs:
                    plan_candidates_x = outputs['plan.fan.candidate_xu'][batch_idx][..., :2].cpu().numpy()  # N, T+1, 6
                    plan_vis.plot_plan_candidates(plan_candidates_x, batch_idx=batch_idx, ax=ax)

                plan_vis.plot_plan_input_batch(batch, batch_idx=batch_idx, ax=ax, legend=False, show=False, close=False)
                plan_vis.plot_plan_result(plan_x, ax=ax)

                # Legend
                plan_vis.legend_unique_labels(ax, loc="best", frameon=True)

                images.append(wandb.Image(
                    fig,
                    caption=f"Batch_idx: {batch_idx}"  # " Pred agent: {batch.agent_name[batch_idx]}"
                ))

            if log_writer:
                log_writer.log({f"{dataset_name}/predictions_viz": images}, step=curr_iter)

            if hyperparams["debug"]:
               print ("Breakpoint here for plot interaction") 
            plt.close("all")
    

                    
    #################################
    #           TRAINING            #
    #################################

    # Start with eval when loading pretrained model.
    if hyperparams["eval_every"] is not None and hyperparams["eval_every"] > 0:
        # if rank == 0 and hyperparams["vis_every"] is not None and hyperparams["vis_every"] > 0:
        #     # run_visualization(0, 0, "eval", eval_dataset)
        #     run_visualization(0, 0, "eval", dataloader=eval_dataloader)
        run_closed_loop_eval(0, 0)
        run_validation(0, 0)

    print (diffstack_module.get_params_summary_text())

    curr_iter: int = 0
    for epoch in range(1, hyperparams['train_epochs'] + 1):
        train_sampler.set_epoch(epoch)
        pbar = tqdm(train_dataloader, ncols=80, unit_scale=hyperparams["world_size"], disable=(rank > 0))
            
            # prof = torch.profiler.profile(
            #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler/tpp_unified'),
            #     record_shapes=True,
            #     profile_memory=True,
            #     with_stack=True
            # )
            # prof.start()
            
            # initialize the timer for the 1st iteration
        step_timer_start = time.time()

        plan_loss_epoch = []
        pred_loss_epoch = []
        plan_metrics_epoch = defaultdict(list)
                    
        batch: AgentBatch
        for batch_idx, batch in enumerate(pbar):
            diffstack_module.set_curr_iter(curr_iter)
            diffstack_module.step_annealers()
                
            # Manually fix seed
            # set_all_seeds(100)

            optimizer.zero_grad(set_to_none=True)

            outputs = diffstack_module(input_wrapper(batch))

            train_loss = outputs["loss"]
            pred_loss = outputs["pred.loss"]
            plan_loss = outputs["plan.loss"]
            plan_metrics = outputs["plan.metrics"]
                
            pbar.set_description(f"Epoch {epoch} L: {train_loss.detach().item():.2f}")
                
            train_loss.backward()

                # Scale down gradients for planning cost
            if hyperparams['train_plan_cost']:
                for param in model_registrar.get_name_match('planner_cost').parameters():
                    if param.grad is None:
                        continue
                    param.grad = param.grad * 0.01  # TODO this has no effect !! but keep it for paper push

                # Clipping gradients.
            if hyperparams['grad_clip'] is not None:
                nn.utils.clip_grad_value_(model_registrar.get_all_but_name_match(['planner_cost']).parameters(), hyperparams['grad_clip'])

            # # Debug gradients
            # if batch_idx == 0:
            #     print ("First batch:")
            #     torch.set_printoptions(precision=10, linewidth=160)
            #     (man_first_history_index,
            #         man_x, man_y, man_x_st_t, man_y_st_t,
            #         man_neighbors_data_st,  # dict of lists. edge_type -> [batch][neighbor]: Tensor(time, statedim). Represetns 
            #         man_neighbors_edge_value,
            #         man_robot_traj_st_t,
            #         man_map, neighbors_future_data, plan_data) = batch.extras["manual_inputs"]
            #     print (man_x_st_t.nan_to_num(0.001).sum().cpu())
            #     print (train_loss)

            #     grads = []
            #     for param in model_registrar.parameters():
            #         if param.grad is None:
            #             continue
            #         grads.append(param.grad.sum())
            #     gradsum = torch.stack(grads).sum()
            #     print ("Grad: ", gradsum)
            #     print (grads)

                # # Check gradients for nans
                # is_grad_nan = False
                # for param in model_registrar.parameters():
                #     if param.grad is None:
                #         continue
                #     is_grad_nan = is_grad_nan or bool(torch.isnan(param.grad).any())
                # if is_grad_nan:
                #     print (batch)
                #     print ("IsNAN:")
                #     print (bool(torch.isnan(train_loss).any()))
                #     print (bool(torch.isnan(plan_loss).any()))
                #     print (bool(torch.isnan(pred_loss).any()))

                #     # Validate inputs
                #     print (batch.agent_fut.isnan().any())
                #     print (any([batch.agent_hist[i, :batch.agent_hist_len[i]].isnan().any() for i in range(256)]))
                #     print (any([batch.agent_hist[i, :batch.agent_hist_len[i]].isnan().any() for i in range(256)]))
                #     print (any([any([batch.neigh_fut[i][j][:batch.neigh_fut_len[i, j]].isnan().any()  for j in  range(batch.num_neigh[i]) ])  for i in range(256)]))
                #     print (any([any([batch.neigh_hist[i][j][:batch.neigh_hist_len[i, j]].isnan().any()  for j in  range(batch.num_neigh[i]) ])  for i in range(256)]))

                #     # Run TPP
                #     node_type = AgentType.VEHICLE
                #     model =  trajectron_module.pred_obj.node_models_dict[node_type.name]
                #     from model.model_utils import ModeKeys
                #     mode = ModeKeys.TRAIN

                #     # encoder
                #     x, x_nr_t, y_e, y_r, y, n_s_t0, dt  = model.obtain_encoded_tensors(mode, batch)
                #     print (any(tensor.isnan().any() for tensor in [x, x_nr_t, y_e, y_r, y, n_s_t0, dt] if tensor is not None))

                #     z, kl = model.encoder(mode, x, y_e)
                #     print (any(tensor.isnan().any() for tensor in [z, kl] if tensor is not None))

                #     log_p_y_xz, y_dist = model.decoder(mode, x, None, y, None, n_s_t0, z, dt, hyperparams['k'], ret_dist=True)
                #     print (any(tensor.isnan().any() for tensor in [log_p_y_xz] if tensor is not None))

                #     # Call forward pass again for an opportunity to debug
                #     for _ in range(20):
                #         train_loss2, plan_loss2, pred_loss2, _, plan_metrics2 = trajectron_module(batch, return_debug=True)
                
            optimizer.step()
                
            # Stepping forward the learning rate scheduler and annealers.
            lr_scheduler.step()
            if rank == 0 and not hyperparams['debug']:
                step_timer_stop = time.time()
                elapsed = step_timer_stop - step_timer_start
                    
                log_writer.log({
                        "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        "train/loss": train_loss.detach().item(),
                        "steps_per_sec": 1 / elapsed,
                        "epoch": epoch,
                        "batch": batch_idx,
                        "global_step": curr_iter,  # TODO kept this for compatibility with old tensorboard based logging
                    }, step=curr_iter, commit=True)

            # Accumulate metrics
            # TODO (pkarkus) remove losses and handle everything inside metrics            
                # TODO (pkarkus) remove losses and handle everything inside metrics            
            # TODO (pkarkus) remove losses and handle everything inside metrics            
            plan_loss_epoch.append(plan_loss.detach().cpu())
            pred_loss_epoch.append(pred_loss.detach().cpu())
            for k, v in plan_metrics.items():
                plan_metrics_epoch[k].append(v.detach().cpu())

            curr_iter += 1
                
            # initialize the timer for the following iteration
            step_timer_start = time.time()

        #     prof.step()
        # Log batch 
        # TODO (pkarkus) simplify this
        # TODO filter by node type

        # Accumulate metrics over epoch
        pred_loss_epoch = torch.stack(pred_loss_epoch, dim=0)
        plan_loss_epoch = torch.stack(plan_loss_epoch, dim=0)
        plan_metrics_epoch = {k: torch.cat(v, dim=0) for k, v in plan_metrics_epoch.items()}
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:  # torch.cuda.is_available() and 
            all_plan_loss_batch = all_gather(plan_loss_epoch)
            plan_loss_epoch = torch.cat(all_plan_loss_batch, dim=0)
            all_pred_loss_batch = all_gather(pred_loss_epoch)
            pred_loss_epoch = torch.cat(all_pred_loss_batch, dim=0)
            all_plan_metrics_batch = all_gather(plan_metrics_epoch)
            plan_metrics_epoch = defaultdict(list)
            for plan_metrics_batch in all_plan_metrics_batch:
                for k, v in plan_metrics_batch.items():
                    plan_metrics_epoch[k].append(v)
            plan_metrics_epoch = {k: torch.cat(v, dim=0) for k, v in plan_metrics_epoch.items()}

        # Log epoch stats
        if rank == 0:
            pred_loss_epoch = pred_loss_epoch.mean()
            plan_loss_epoch = plan_loss_epoch.mean()

            if log_writer:
                log_writer.log({
                    f"train/epoch_pred_loss": pred_loss_epoch.detach().item()
                    }, step=curr_iter, commit=False)
                log_writer.log({
                    f"train/epoch_plan_loss": plan_loss_epoch.mean().detach().item()
                    }, step=curr_iter, commit=False)

            print (f"Epoch {epoch} pred_loss {pred_loss_epoch} plan_loss {plan_loss_epoch}.")

            if "fan_valid" in plan_metrics_epoch:
                valid_filter = plan_metrics_epoch["fan_valid"]
                for k, v in plan_metrics_epoch.items():
                    v_mean = v.float().mean().detach().item()
                    v_mean_valid = v[valid_filter].float().mean().detach().item()

                    if log_writer:
                        log_writer.log({
                            f"train/epoch_{k}": v_mean,
                            f"train/epoch_{k}_valid": v_mean_valid,
                            }, step=curr_iter, commit=False)

                    print (f"{k}: {v_mean} {v_mean_valid}")

            print (diffstack_module.get_params_summary_text())

        del plan_loss_epoch
        del pred_loss_epoch
        del plan_metrics_epoch

        # prof.stop()
        # raise
        if hyperparams['lr_step'] != 0:
            step_scheduler.step()


        #################################
        #           VALIDATION          #
        #################################
        if hyperparams["eval_every"] is not None and hyperparams["eval_every"] > 0 and epoch % hyperparams["eval_every"] == 0 and epoch > 0:
            run_validation(epoch, curr_iter)

        if rank == 0 and (hyperparams["save_every"] is not None and not hyperparams["debug"] and epoch % hyperparams["save_every"] == 0):
            model_registrar.save_models(epoch)

        #################################
        #        VISUALIZATION          #
        #################################
        if rank == 0 and (hyperparams["planner"] not in ["", "none"]) and (hyperparams["vis_every"] is not None and hyperparams["vis_every"] > 0 and epoch % hyperparams["vis_every"] == 0 and epoch > 0):
            # run_visualization(epoch, curr_iter, "eval", eval_dataset)
            run_visualization(epoch, curr_iter, "eval", dataloader=eval_dataloader)

        # Waiting for process 0 to be done its evaluation and visualization.
        if torch.distributed.is_initialized() and torch.cuda.is_available():
            torch.distributed.barrier()

    return model_dir


def prepare_logging(rank, hyperparams):
    # Logging
    log_writer = None
    model_dir = None
    if not hyperparams["debug"]:
        # Create the log and model directory if they're not present.
        model_dir = os.path.join(hyperparams["log_dir"],
                                 hyperparams["experiment"] + time.strftime('-%d_%b_%Y_%H_%M_%S', time.localtime()))
        hyperparams["logdir"] = model_dir

        if rank == 0:
            pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

            # Save config to model directory
            with open(os.path.join(model_dir, 'config.json'), 'w') as conf_json:
                json.dump(hyperparams, conf_json)

            # wandb.tensorboard.patch(root_logdir=model_dir, pytorch=True)

            # WandB init. Put it in a loop because it can fail on ngc.
            for _ in range(10):
                try:
                    log_writer = wandb.init(
                        project="debug" if hyperparams["debug"] else hyperparams["experiment"], name=f"{hyperparams['experiment']}",
                        config=hyperparams, mode="offline" if hyperparams["debug"] else "online")  # sync_tensorboard=True, 
                except:
                    continue
                break
            else:
                raise ValueError("Could not connect to wandb")
            
            artifact = wandb.Artifact('logdir', type='path')
            artifact.add_dir(model_dir)
            wandb.log_artifact(artifact)

            # log_writer = SummaryWriter(log_dir=model_dir)
            print (f"Log path: {model_dir}")
    return log_writer, model_dir


if __name__ == '__main__':

    if torch.distributed.is_torchelastic_launched():
        local_rank = int(os.environ["LOCAL_RANK"])
        initialize_torch_distributed(local_rank)

        print(
            f"[{os.getpid()}]: world_size = {torch.distributed.get_world_size()}, "
            + f"rank = {torch.distributed.get_rank()}, backend={torch.distributed.get_backend()}, "
            + f"port = {os.environ['MASTER_PORT']} \n", end=''
        )        
    else:
        local_rank = 0 

    log_dir = train(local_rank, args)

    wandb.finish()
