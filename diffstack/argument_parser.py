import argparse
import json
import torch
import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--conf",
                    help="path to json config file for hyperparameters",
                    default="./config/diffstack_default.json",
                    type=str)

parser.add_argument("--experiment",
                    help="name of experiment for wandb",
                    type=str,
                    default='diffstack-def')

parser.add_argument("--debug",
                    help="disable all disk writing processes.",
                    action='store_true')

parser.add_argument("--preprocess_workers",
                    help="number of processes to spawn for preprocessing",
                    type=int,
                    default=0)

parser.add_argument('--seed',
                    help='manual seed to use, default is 123',
                    type=int,
                    default=123)

parser.add_argument('--device',
                    help='what device to perform training on',
                    type=str,
                    default='cuda:0')  

# Data Parameters
parser.add_argument("--data_source",
                    help="Specifies the source of data [trajdata, trajdata-scene, cache]",
                    type=str,
                    default='trajdata')

parser.add_argument("--data_loc_dict",
                    help="JSON dict of dataset locations",
                    type=str,
                    default='{"nusc_mini": "./data/nuscenes"}')

parser.add_argument("--cached_data_dir",
                    help="What dir to look in for cached data when data_source==cache",
                    type=str,
                    default='./data/cached_data')

parser.add_argument('--train_data',
                    help='name of data split to use for training',
                    type=str,
                    default="nusc_mini-mini_train")  

parser.add_argument("--eval_data",
                    help="name of data split to use for evaluation",
                    type=str,
                    default='nusc_mini-mini_val')

parser.add_argument("--log_dir",
                    help="what dir to save training information (i.e., saved models, logs, etc)",
                    type=str,
                    default='./experiments/')

parser.add_argument("--trajdata_cache_dir",
                    help="location of the unified dataloader cache",
                    type=str,
                    default='./cache/unified_data_cache')
 
parser.add_argument('--rebuild_cache',
                    help="Rebuild trajdata cache files.",
                    action='store_true')   

# Training and eval parameters
parser.add_argument('--load',
                    help='Load pretrained model from checkpoint file if not empty.',
                    type=str,
                    default="")  

parser.add_argument('--batch_size',
                    help='training batch size',
                    type=int,
                    default=256)

parser.add_argument('--eval_batch_size',
                    help='evaluation batch size',
                    type=int,
                    default=256)

parser.add_argument("--learning_rate",
                    help="initial learning rate",
                    type=float,
                    default=0.003)

parser.add_argument("--map_enc_learning_rate",
                    help="map encoder learning rate for trajectron",
                    type=float,
                    default=0.00003)

parser.add_argument("--lr_step",
                    help="number of epochs after which to step down the LR by 0.1, the default (0) is no step downs",
                    type=int,
                    default=0)

parser.add_argument("--grad_clip",
                    help="the maximum magnitude of gradients (enforced by clipping)",
                    type=float,
                    default=1.0)

parser.add_argument("--train_epochs",
                    help="number of epochs to train for",
                    type=int,
                    default=20)

parser.add_argument('--eval_every',
                    help='how often to evaluate during training, never if None',
                    type=int,
                    default=1)

parser.add_argument('--vis_every',
                    help='how often to visualize during training, never if None',
                    type=int,
                    default=0)

parser.add_argument('--save_every',
                    help='how often to save during training, never if None',
                    type=int,
                    default=1)

parser.add_argument('--log_histograms',
                    help="Log historgrams during training.",
                    action='store_true')

parser.add_argument('--cl_trajlen',
                    help='Length of trajectories for closed loop evaluation in discrete timesteps. No closed loop evaluation if negative.',
                    type=int,
                    default=-1)

# Stack parameters
parser.add_argument('--predictor',
                    help='Choose prediction model [tpp, constvel]',
                    type=str,
                    default="tpp_origin")

parser.add_argument('--planner',
                    help='Choice of planner [none, mpc, fan, fan_mpc, cvae]',
                    type=str,
                    default="fan_mpc")

parser.add_argument("--history_sec",
                    help="required agent history (in seconds)",
                    type=float,
                    default=4.0)

parser.add_argument("--prediction_sec",
                    help="prediction horizon (in seconds)",
                    default=3.0,
                    type=float)

parser.add_argument('--plan_dt',
                    help='Custom delta t for planning. Will be the same as prediction dt if not defined or nonpositive.',
                    type=float,
                    default=0.)

parser.add_argument('--plan_cost',
                    help='Replaces the cost function for planning if specified [[empty], corl_default, corl_default_angle_fix]',
                    type=str,
                    default="corl_default_angle_fix")

parser.add_argument('--plan_init',
                    help='Options for initializing the planner: fitted, gtplan, zero.',
                    type=str,
                    default="zero")

parser.add_argument('--plan_lqr_eps',
                    help='Maximum change in norm of control vector for iLQR to be considered converged.',
                    type=float,
                    default=0.01)

parser.add_argument('--plan_loss',
                    help='Options for the planning loss [mse, joint_hcost2, class_hcost2, hcost]',
                    type=str,
                    default="joint_hcost2")

parser.add_argument('--no_plan_train',
                    help="Disable planning during training.",
                    action='store_true')

parser.add_argument('--no_train_pred',
                    help="Disable updating the prediction model during training.",
                    action='store_true')

parser.add_argument('--train_plan_cost',
                    help="Train the planners cost function.",
                    action='store_true')   

parser.add_argument('--bias_predictions',
                    help="Add a bias to prediction targets if True.",
                    action='store_true')   

parser.add_argument('--pred_loss_scaler',
                    help='Scaler for prediction loss.',
                    type=float,
                    default=1.0)

parser.add_argument('--plan_loss_scaler',
                    help='Scaler for planning loss when added to prediction loss.',
                    type=float,
                    default=100.0)

parser.add_argument('--plan_loss_scaler2',
                    help='Scaler for the second planning loss term (when used).',
                    type=float,
                    default=10.0)                    

parser.add_argument('--pred_loss_weights',
                    help='Custom scheme to weight prediction losses across samples in a batch [none, dist, grad]',
                    type=str,
                    default="none")

parser.add_argument('--pred_loss_temp',
                    help='Temperature parameter for prediction loss weighting scheme.',
                    type=float,
                    default=1.0)

parser.add_argument('--cost_grad_scaler',
                    help='Scaler for the gradient of cost weights when training the planning cost.',
                    type=float,
                    default=0.001)

parser.add_argument('--plan_loss_scale_start',
                    help='Epoch where to start (linearly) increasing planning loss scaler.',
                    type=int,
                    default=-1)

parser.add_argument('--plan_loss_scale_end',
                    help='Epoch where to end (linearly) increasing planning loss scaler.',
                    type=int,
                    default=-1)                    

# Trajectron++
parser.add_argument('--K',
                    help='how many CVAE discrete latent modes to have in the model',
                    type=int,
                    default=25)

parser.add_argument('--k_eval',
                    help='how many samples to take during evaluation',
                    type=int,
                    default=25)

parser.add_argument('--pred_ego_indicator',
                    help="The type of ego indicator to use in input to predictor [none, most_relevant].",
                    type=str,
                    default='most_relevant')

parser.add_argument("--dynamic_edges",
                    help="whether to use dynamic edges or not, options are 'no' and 'yes'",
                    type=str,
                    default='no')

parser.add_argument("--edge_state_combine_method",
                    help="the method to use for combining edges of the same type",
                    type=str,
                    default='sum')

parser.add_argument("--edge_influence_combine_method",
                    help="the method to use for combining edge influences",
                    type=str,
                    default='attention')

parser.add_argument('--edge_addition_filter',
                    nargs='+',
                    help="what scaling to use for edges as they're created",
                    type=float,
                    default=[0.25, 0.5, 0.75, 1.0]) # We don't automatically pad left with 0.0, if you want a sharp
                                                    # and short edge addition, then you need to have a 0.0 at the
                                                    # beginning, e.g. [0.0, 1.0].

parser.add_argument('--edge_removal_filter',
                    nargs='+',
                    help="what scaling to use for edges as they're removed",
                    type=float,
                    default=[1.0, 0.0])  # We don't automatically pad right with 0.0, if you want a sharp drop off like
                                         # the default, then you need to have a 0.0 at the end.

parser.add_argument('--incl_robot_node',
                    help="whether to include a robot node in the graph or simply model all agents",
                    action='store_true')

parser.add_argument('--map_encoding',
                    help="Whether to use map encoding or not",
                    action='store_true')

parser.add_argument('--augment_input_noise',
                    help="Standard deviation of Gaussian noise to add the inputs during training, not performed if 0.0",
                    type=float,
                    default=0.0)

parser.add_argument('--no_edge_encoding',
                    help="Whether to use neighbors edge encoding",
                    action='store_true')


args = parser.parse_args()


def get_hyperparams(args):
    # Load hyperparameters from json
    if not os.path.exists(args.conf):
        print('Config json not found!')
    with open(args.conf, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    # Add all arguments as hyperparams
    hyperparams.update({k: v for k, v in vars(args).items() if v is not None})
    if torch.distributed.is_initialized():
        hyperparams['world_size'] = torch.distributed.get_world_size()  # number of torch workers
    else:
        hyperparams['world_size'] = 1

    # Special arguments
    hyperparams['plan_dt'] = (args.plan_dt if args.plan_dt > 0. else hyperparams['dt'])
    hyperparams['edge_encoding'] = not args.no_edge_encoding
    hyperparams['plan_train'] = not args.no_plan_train
    hyperparams['train_pred'] = not args.no_train_pred

    # Distributed LR Scaling
    hyperparams['learning_rate'] *= hyperparams['world_size']

    return hyperparams


def print_hyperparams_summary(hyperparams):
    print('-----------------------')
    print('| PARAMETERS |')
    print('-----------------------')
    print('| Experiment: %s' % hyperparams["experiment"])
    print('| Batch Size: %d' % hyperparams["batch_size"])
    print('| Eval Batch Size: %d' % hyperparams["eval_batch_size"])
    print('| Device: %s %d' % (hyperparams["device"], hyperparams["world_size"]))
    print('| Learning Rate: %s' % hyperparams['learning_rate'])
    print('| Learning Rate Step Every: %s' % hyperparams["lr_step"])
    print('| Max History: %ss' % hyperparams['history_sec'])
    print('| Max Future: %ss' % hyperparams['prediction_sec'])
    print('| Args: %s' % " ".join(sys.argv[1:]))
    print('-----------------------')
