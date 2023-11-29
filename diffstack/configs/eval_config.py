import numpy as np
from copy import deepcopy

from diffstack.configs.config import Dict


class SimEvaluationConfig(Dict):
    def __init__(self):
        super(SimEvaluationConfig, self).__init__()
        self.name = None
        self.env = "nusc"  
        self.dataset_path = None
        self.eval_class = ""
        self.seed = 0
        self.num_scenes_per_batch = 4
        self.num_scenes_to_evaluate = 100

        self.num_episode_repeats = 3
        self.start_frame_index_each_episode = None  # if specified, should be the same length as num_episode_repeats
        self.seed_each_episode = None  # if specified, should be the same length as num_episode_repeats

        self.ego_only = False
        self.agent_eval_class = None

        self.ckpt_root_dir = "checkpoints/"
        self.experience_hdf5_path = None
        self.results_dir = "results/"

        self.ckpt.policy.ngc_job_id = None
        self.ckpt.policy.ckpt_dir = None
        self.ckpt.policy.ckpt_key = None

        self.ckpt.planner.ngc_job_id = None
        self.ckpt.planner.ckpt_dir = None
        self.ckpt.planner.ckpt_key = None

        self.ckpt.predictor.ngc_job_id = None
        self.ckpt.predictor.ckpt_dir = None
        self.ckpt.predictor.ckpt_key = None

        self.ckpt.cvae_metric.ngc_job_id = None
        self.ckpt.cvae_metric.ckpt_dir = None
        self.ckpt.cvae_metric.ckpt_key = None

        self.ckpt.occupancy_metric.ngc_job_id = None
        self.ckpt.occupancy_metric.ckpt_dir = None
        self.ckpt.occupancy_metric.ckpt_key = None

        self.policy.mask_drivable = True
        self.policy.num_plan_samples = 50
        self.policy.num_action_samples = 10
        self.policy.pos_to_yaw = True
        self.policy.yaw_correction_speed = 1.0
        self.policy.diversification_clearance = None
        self.policy.sample = True


        self.policy.cost_weights.collision_weight = 10.0
        self.policy.cost_weights.lane_weight = 1.0
        self.policy.cost_weights.likelihood_weight = 0.0  # 0.1
        self.policy.cost_weights.progress_weight = 0.0  # 0.005

        self.metrics.compute_analytical_metrics = True
        self.metrics.compute_learned_metrics = False

        self.perturb.enabled = False
        self.perturb.OU.theta = 0.8
        self.perturb.OU.sigma = [0.0, 0.1,0.2,0.5,1.0,2.0,4.0]
        self.perturb.OU.scale = [1.0,1.0,0.2]

        self.rolling_perturb.enabled = False
        self.rolling_perturb.OU.theta = 0.8
        self.rolling_perturb.OU.sigma = 0.5
        self.rolling_perturb.OU.scale = [1.0,1.0,0.2]

        self.occupancy.rolling = True
        self.occupancy.rolling_horizon = [5,10,20]

        self.cvae.rolling = True
        self.cvae.rolling_horizon = [5,10,20]

        self.nusc.eval_scenes = np.arange(100).tolist()
        self.nusc.n_step_action = 5
        self.nusc.num_simulation_steps = 200
        self.nusc.skip_first_n = 0


        self.adjustment.random_init_plan=True
        self.adjustment.remove_existing_neighbors = False
        self.adjustment.initial_num_neighbors = 4
        self.adjustment.num_frame_per_new_agent = 20

    def clone(self):
        return deepcopy(self)

class EvaluationConfig(Dict):
    def __init__(self):
        super(EvaluationConfig, self).__init__()
        self.name = None
        self.env = "nusc"  
        self.dataset_path = None
        self.eval_class = ""
        self.seed = 0
        self.ckpt_root_dir = "checkpoints/"
        self.ckpt.dir = None
        self.ckpt.ngc_job_id = None
        self.ckpt.ckpt_dir = None
        self.ckpt.ckpt_key = None
        
        self.eval.batch_size = 100
        self.eval.num_steps = None
        self.eval.num_data_workers = 8
        self.log_image_frequency=None
        self.log_all_image = False
        
        self.trajdata_source_root = "nusc_trainval"
        self.trajdata_source_eval = "val"
        

class TrainTimeEvaluationConfig(SimEvaluationConfig):
    def __init__(self):
        super(TrainTimeEvaluationConfig, self).__init__()

        self.num_scenes_per_batch = 4
        self.nusc.eval_scenes = np.arange(0, 100, 10).tolist()
        self.policy.sample = False
