import math

from diffstack.configs.base import AlgoConfig


class BehaviorCloningConfig(AlgoConfig):
    def __init__(self):
        super(BehaviorCloningConfig, self).__init__()
        self.eval_class = "BC"

        self.name = "bc"
        self.model_architecture = "resnet18"
        self.map_feature_dim = 256
        self.history_num_frames = 8
        self.history_num_frames_ego = 8
        self.history_num_frames_agents = 8
        self.future_num_frames = 6
        self.step_time = 0.5
        self.render_ego_history = False

        self.decoder.layer_dims = ()
        self.decoder.state_as_input = True

        self.dynamics.type = "Unicycle"
        self.dynamics.max_steer = 0.5
        self.dynamics.max_yawvel = math.pi * 2.0
        self.dynamics.acce_bound = (-10, 8)
        self.dynamics.ddh_bound = (-math.pi * 2.0, math.pi * 2.0)
        self.dynamics.max_speed = 40.0  # roughly 90mph

        self.spatial_softmax.enabled = False
        self.spatial_softmax.kwargs.num_kp = 32
        self.spatial_softmax.kwargs.temperature = 1.0
        self.spatial_softmax.kwargs.learnable_temperature = False

        self.loss_weights.prediction_loss = 1.0
        self.loss_weights.goal_loss = 0.0
        self.loss_weights.collision_loss = 0.0
        self.loss_weights.yaw_reg_loss = 0.001

        self.optim_params.policy.learning_rate.initial = 1e-3  # policy learning rate
        self.optim_params.policy.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.policy.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params.policy.regularization.L2 = 0.00  # L2 regularization strength
        self.checkpoint.enabled = False
        self.checkpoint.path = None


class CTTConfig(AlgoConfig):
    def __init__(self):
        super(CTTConfig, self).__init__()
        self.name = "CTT"
        self.step_time = 0.25
        self.history_num_frames = 4
        self.future_num_frames = 12

        self.n_embd = 256
        self.n_head = 8
        self.use_rpe_net = True
        self.PE_mode = "RPE"  # "RPE" or "PE"

        self.enc_nblock = 3
        self.dec_nblock = 3

        self.encoder.attn_pdrop = 0.05
        self.encoder.resid_pdrop = 0.05
        self.encoder.pooling = "attn"
        self.encoder.edge_scale = 20
        self.encoder.edge_clip = [-4, 4]
        self.encoder.mode_embed_dim = 64
        self.encoder.jm_GNN_nblock = 2
        self.encoder.num_joint_samples = 30
        self.encoder.num_joint_factors = 6
        self.encoder.null_lane_mode = True

        self.edge_dim.a2a = 14
        self.edge_dim.a2l = 12
        self.edge_dim.l2a = 12
        self.edge_dim.l2l = 16
        self.a2l_edge_type = "attn"
        self.a2l_n_embd = 64

        self.attn_ntype.a2a = 2
        self.attn_ntype.a2l = 1
        self.attn_ntype.l2l = 2

        self.lane_GNN_num_layers = 4
        self.homotpy_GNN_num_layers = 4

        self.hist_lane_relation = "LaneRelation"
        self.fut_lane_relation = "SimpleLaneRelation"
        self.classify_a2l_4all_lanes = (
            False  # Alternative is 4all modes, find resulting lane
        )

        self.decoder.arch = "mlp"
        self.decoder.lstm_hidden_size = 128
        self.decoder.mlp_hidden_dims = [128, 256]
        self.decoder.traj_dim = 4  # x,y,v,yaw
        self.decoder.num_layers = 2
        self.decoder.dyn.vehicle = "unicycle"
        self.decoder.dyn.pedestrian = "DI_unicycle"
        self.decoder.attn_pdrop = 0.05
        self.decoder.resid_pdrop = 0.05
        self.decoder.pooling = "attn"
        self.decoder.decode_num_modes = 5
        self.decoder.AR_step_size = 1
        self.decoder.GNN_enabled = False
        self.decoder.AR_update_mode = None
        self.decoder.dec_rounds = 5

        self.num_lane_pts = 30

        self.loss_weights.marginal_lm_loss = 5.0
        self.loss_weights.marginal_homo_loss = 5.0
        self.loss_weights.joint_prob_loss = 5.0
        self.loss_weights.xy_loss = 4.0
        self.loss_weights.yaw_loss = 1.0
        self.loss_weights.lm_consistency_loss = 5.0
        self.loss_weights.homotopy_consistency_loss = 5.0
        self.loss_weights.yaw_dev_loss = 30.0
        self.loss_weights.y_dev_loss = 5.0
        self.loss_weights.l2_reg = 0.1
        self.loss_weights.coll_loss = 0.1
        self.loss_weights.acce_reg_loss = 0.04
        self.loss_weights.steering_reg_loss = 0.1
        self.loss_weights.input_violation_loss = 20.0
        self.loss_weights.jerk_loss = 0.05

        self.loss.lm_margin_offset = 0.2

        self.weighted_consistency_loss = False
        self.LR_sample_hack = True

        self.scene_centric = True

        self.max_joint_cardinality = 5

        self.optim_params.policy.learning_rate.initial = 1e-4  # policy learning rate
        self.optim_params.policy.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.policy.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params.policy.regularization.L2 = 0.00  # L2 regularization strength
