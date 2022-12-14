import torch
import numpy as np

from typing import Dict, Iterable, Optional, Union, Any, Tuple
from diffstack.modules.cost_functions.linear_base_cost import LinearBaseCost
from diffstack.utils.utils import pt_rbf, angle_wrap

# Traceable torch functions are meant to be used when tracing computation graph
# for faster autodifferentiation when computing quadratic cost approximations.
# However, in the default LinearCost1 cost every component has analytic gradient 
# implmenentations, so these traceable functions are not useful, hence replacing them
# with their original torch counterpart.
tracable_norm = torch.linalg.norm
tracable_rbf = pt_rbf


class LinearCost1(LinearBaseCost):
    """Linear cost function used in the CoRL 2022 paper. 

    The cost is a linear combination of the following terms (in this order):
    - lane lateral cost
    - lane heading cost
    - goal cost
    - control cost
    - collision cost
    """
    theta_dim = 5

    def forward(self, xu, cost_inputs, keep_components=False):
        """Compute cost."""
        gt_neighbors_batch, mus_batch, probs_batch, goal_batch, lanes, _ = cost_inputs        
        return self._compute_cost(xu, self.theta, gt_neighbors_batch, mus_batch, probs_batch, goal_batch, lanes, rbf_scale=self.rbf_scale_long, keep_components=keep_components)

    def approx_quadratic(self, x, u, cost_inputs, diff=True):
        """Compute the quadratic approximation of the cost."""
        gt_neighbors_batch, mus_batch, probs_batch, goal_batch, lanes, _ = cost_inputs

        return self._approx_quadratic(x, u, self.theta, gt_neighbors_batch, mus_batch, probs_batch, goal_batch, lanes, rbf_scale=self.rbf_scale_long, diff=diff)

    @classmethod
    def _compute_cost(
            cls,
            xu: torch.Tensor,   # (T+1, x_dims+u_dims) or (T+1, b, x_dims+u_dims)
            theta: torch.Tensor,
            gt_neighbors: torch.Tensor,  # (N-1 or 0, T, K, 2)
            mus: torch.Tensor,  # (1 or N, T, K, 2)
            probs: torch.Tensor,  # (1 or N, T, K)
            goal: torch.Tensor,  # (2, )
            lanes: torch.Tensor,
            rbf_scale: Union[float, torch.Tensor] = 2.0,
            keep_components: bool = False,
            ):

        # Cost terms
        cost_terms = cls._compute_cost_terms(xu, gt_neighbors, mus, probs, goal, lanes, rbf_scale=rbf_scale)

        # Theta
        if xu.ndim == 3:
            theta_vec = theta.unsqueeze(1).unsqueeze(0)  # 1, theta_dim, 1
        elif xu.ndim == 2:
            theta_vec = theta.unsqueeze(1)  # theta_dim, 1
        else:
            raise ValueError

        # Weighted sum or weighted components
        if keep_components:
            return (cost_terms * theta_vec.transpose(-1, -2))
        else:
            cost = torch.matmul(cost_terms, theta_vec).squeeze(-1)
            return cost  # (T, ) or (T, b)

    @staticmethod
    def _compute_cost_terms(
            xu: torch.Tensor,   # (T+1, x_dims+u_dims) or (T+1, b, xu)
            gt_neighbors: torch.Tensor,  # (N-1 or 0, T, K, 2) or list of same for batch
            mus: torch.Tensor,  # (1 or N, T, K, 2) or list of same for batch
            probs: torch.Tensor,  # (1 or N, T, K)  or list of same for batch
            goal: torch.Tensor,  # (2, ) or (b, 2)
            lanes: torch.Tensor,
            rbf_scale: float,
            ):  #(T+1, 3, ) or (T+1, b, 3)

        x, u = torch.split(xu, (4, 2), dim=-1)  # x, y, orient, vel, d_orient, acc

        # Deal with different time resolutions for prediction and planning
        predh = lanes.shape[0]-1
        planh = x.shape[0]-1
        if planh != predh:
            assert planh % predh == 0 and planh > predh
            num_repeat = planh // predh
            gt_neighbors = None if gt_neighbors is None else torch.repeat_interleave(gt_neighbors, num_repeat, dim=1)
            mus = None if mus is None else torch.repeat_interleave(mus, num_repeat, dim=1)
            lanes = torch.repeat_interleave(lanes, num_repeat, dim=0)[num_repeat-1:]

        assert lanes.shape[:-1] == x.shape[:-1]
        assert len(goal.shape) == len(x.shape)-1

        ego_lane_lat = torch.square(x[..., :2] - lanes[..., :2]).sum(dim=-1)
        ego_lane_heading = torch.square(angle_wrap(x[..., 2] - lanes[..., 2]))
        # ego_lane_heading = torch.square(x[..., 2] - lanes[..., 2])
        ego_goal = torch.cat((torch.zeros_like(ego_lane_heading)[:-1],
                              torch.square(x[-1, ..., :2] - goal).sum(dim=-1).unsqueeze(0)), dim=0) * 0.1
        control_cost = torch.square(u).sum(dim=-1)

        collision_reward = LinearCost1._collision_reward(
            mus, probs, x, gt_neighbors=gt_neighbors, rbf_scale=rbf_scale)

        cost_terms = [
            ego_lane_lat, 
            ego_lane_heading,
            ego_goal,
            control_cost,
            -collision_reward]

        cost_terms = torch.stack(cost_terms, dim=-1)  # t, (b), theta_dim            
        return cost_terms

    @staticmethod
    def _collision_reward(pred_mus: torch.Tensor, pred_probs: torch.Tensor, ego_x: torch.Tensor, gt_neighbors: torch.Tensor = None, rbf_scale = 2.0, return_grad: bool = False):
        # The collision term (with analyitic gradient) is only implemented for a single sample
        # so for batched inputs we iterate over the batch. This is highly inefficient. 
        is_batched = (len(ego_x.shape) == 3)
        if is_batched:
            reward_outputs = []
            for b_i in range(ego_x.shape[1]):                           
                reward_outputs.append(LinearCost1._collision_reward_single_analytic_grad(  # recursive call to itslef
                    pred_mus[:, :, b_i], pred_probs[:, b_i], ego_x[:, b_i], 
                    gt_neighbors=(None if gt_neighbors is None else gt_neighbors[:, :, b_i]),
                    rbf_scale=rbf_scale,
                    return_grad=return_grad))

            if return_grad:
                # TODO refactor with zip
                prediction_reward = torch.stack([output[0] for output in reward_outputs], dim=1)
                gradients = torch.stack([output[1] for output in reward_outputs], dim=1)
                hessians = torch.stack([output[2] for output in reward_outputs], dim=1)
                return prediction_reward, gradients, hessians                
            else:
                prediction_reward = torch.stack(reward_outputs, dim=1)
                return prediction_reward
        else:      
            return LinearCost1._collision_reward_single_analytic_grad(pred_mus, pred_probs, ego_x, gt_neighbors, rbf_scale=rbf_scale, return_grad=return_grad)

    @staticmethod
    def _collision_reward_single_analytic_grad(pred_mus: torch.Tensor, pred_probs: torch.Tensor, x: torch.Tensor, gt_neighbors: torch.Tensor = None, rbf_scale = 2.0, return_grad: bool = False):
        """Computes the collision cost and optionally its first and second order gradients.
        
        Here the gradients are analytically computed. The experimental_costs.py provides alternative implementations 
        using autodiff that is more general but much less efficient. There have also been attempts to trace the autodiff function, i.e.
        create the computation graph for the gradient only once. This does speed up the autodiff variant but the analytic implementation
        is still faster.
        """
        # For now only support single sample, no batch.
        assert len(x.shape) == 2
        assert len(pred_mus.shape) == 4
        assert len(pred_probs.shape) == 2

        x_mu_delta = x[1:pred_mus.shape[1]+1, :2].unsqueeze(0).unsqueeze(2) - torch.nan_to_num(pred_mus, nan=1e6)
        comp_dists = tracable_norm(x_mu_delta, dim=-1)  # Distance from predictions (nodes, predhorizon, K)

        # 2) Keep time, produce one value for each time step.
        expected_dist = torch.sum(comp_dists * pred_probs.unsqueeze(1), dim=-1)  # Factoring in traj. probabilities -> expected closest encounter (nodes, predhorizon)
        
        # Add in gt
        if gt_neighbors is not None:
            x_gt_delta = x[1:gt_neighbors.shape[1]+1, :2].unsqueeze(0) - torch.nan_to_num(gt_neighbors, nan=1e6)
            gt_comp_dists = tracable_norm(x_gt_delta, dim=-1)  # Distance from predictions (nodes, predhorizon)
            expected_dist = torch.cat((expected_dist, gt_comp_dists), axis=0)

        # Instead of closest agent, add distance term for all agents
        expected_dist = expected_dist.unsqueeze(-1)  # (N, T, 1)
        full_cost = -tracable_rbf(expected_dist, scale=rbf_scale)  # (N, T, )
        ret = full_cost.sum(dim=0)  # sum over agents (T, )
        ret = torch.cat((torch.zeros((1, ), dtype=ret.dtype, device=ret.device), ret), dim=0) # extend to (T+1, )

        if return_grad:
            cost_over_scale = full_cost.unsqueeze(-1) / rbf_scale  # expected_dist (N, T, 1). Denoted F over a in symbolic_diff.
            pred_probs_ext = pred_probs.unsqueeze(1).unsqueeze(-1) # N_pred, 1, K, 1. Denoted p in symbolic_diff.
            comp_dists_ext = comp_dists.unsqueeze(-1)  # N_pred, T, K, 1. Denoted d in symbolic_diff.
            # expected_dist_pred_agents: This is sum of d over K weighted with p. Denoted s_dp on paper.
            expected_dist_pred_agents = expected_dist[:x_mu_delta.shape[0]] # N_pred, T, 1. 
            # precompute common terms
            # x_mu_delta: x-x_ak  # N_pred, T, K, 2
            x_mu_delta_over_comp_dists = x_mu_delta / comp_dists_ext  # (x-x_ak)/d
            x_mu_delta_over_comp_dists_squared = x_mu_delta_over_comp_dists.square()  # ((x-x_ak)/d)^2
            pred_probs_over_comp_dists = pred_probs_ext / comp_dists_ext

            # --- Gradient
            # Gradient for predicted agents
            grad_pred = x_mu_delta * pred_probs_over_comp_dists  # N_pred, T, K, 2
            grad_pred = expected_dist_pred_agents * grad_pred.sum(-2)  # sum over K, remains: N_pred, T, 2                    
            if gt_neighbors is not None:
                # Gradient for gt agents
                grad_gt = x_gt_delta  # N_gt, T, 2
                # Gradient combine predicted and gt
                grad = torch.cat((grad_pred, grad_gt), axis=0)  # N, T, 2
            else:
                grad = grad_pred

            grad = -cost_over_scale * grad  # N, T, 2
            grad = grad.sum(0)  # sum over agents (T, 2)
            grad = torch.cat((torch.zeros((1, 2), dtype=grad.dtype, device=grad.device), grad), dim=0) # extend to (T+1, )

            # --- Hessian for diagonals H11 and H22
            # Hessian diagonals for predicted agents
            hess_d1 = pred_probs_over_comp_dists * (x_mu_delta_over_comp_dists_squared - 1.)  #  N_pred, T, K, 2
            hess_d1 = expected_dist_pred_agents * hess_d1.sum(-2)  # sum over K, remains: N_pred, T, 2

            hess_d2 = pred_probs_over_comp_dists * x_mu_delta  # N_pred, T, K, 2
            hess_d2 = hess_d2.sum(-2)  # sum over K, remains N_pred, T, 2
            shared_term1 = (expected_dist_pred_agents.square() / rbf_scale - 1.) 
            hess_d2 = hess_d2.square() * shared_term1  # N_pred, T, 2

            hess_d_pred = (hess_d1 + hess_d2)  # N_pred, T, 2

            # --- Hessian for antidiagonals H12 == H21
            # Hessian antidiagonals for predicted agents
            hess_a1 = x_mu_delta.prod(dim=-1, keepdim=True)   # (x-x_a)(y-y_a), remains N_pred, T, K, 1
            hess_a1 = hess_a1 * pred_probs_over_comp_dists / comp_dists_ext.square()  # N_pred, T, K, 1
            hess_a1 = expected_dist_pred_agents * hess_a1.sum(-2)  # sum over K, remains N_pred, T, 1

            hess_a2 = pred_probs_over_comp_dists * x_mu_delta  # N_pred, T, K, 2
            hess_a2 = hess_a2.sum(-2)  # sum over K, remains: N_pred, T, 2
            hess_a2 = hess_a2.prod(dim=-1, keepdim=True)  # prod over xy, remains: N_pred, T, 1
            hess_a2 = hess_a2 * shared_term1

            hess_a_pred = (hess_a1 + hess_a2)  # N_pred, T, 2

            # ---- Combine pred and gt
            if gt_neighbors is not None:
                # Hessian diagonals for gt agents
                hess_d_gt = (x_gt_delta.square()/rbf_scale - 1.)  # N_gt, T, 2

                # combine predicted and gt agents
                hess_d = cost_over_scale * torch.cat((hess_d_pred, hess_d_gt), axis=0)  # N, T, 2

                # Hessian antidiagonals for gt agents
                hess_a_gt = x_gt_delta.prod(dim=-1, keepdim=True)/rbf_scale  # N_gt, T, 1

                # combine predicted and gt agents
                hess_a = cost_over_scale * torch.cat((hess_a_pred, hess_a_gt), axis=0)  # N, T, 1

            else:
                hess_d = cost_over_scale * hess_d_pred  # N, T, 2
                hess_a = cost_over_scale * hess_a_pred  # N, T, 1

            # Build Hessian matrix from H11, H22, H12
            hess_11, hess_22 = torch.split(hess_d, (1, 1), dim=-1)
            hess_12 = hess_a
            hess_21 = hess_a

            hess = torch.stack((
                torch.cat((hess_11, hess_12), dim=-1),  # first column
                torch.cat((hess_21, hess_22), dim=-1)),  # second column
                dim=-1)  # N, T, 2, 2
            
            hess = hess.sum(0)  # sum over agents (T, 2)
            hess = torch.cat((torch.zeros((1, 2, 2), dtype=hess.dtype, device=hess.device), hess), dim=0) # extend to (T+1, )

            return ret, grad, hess

        return ret

    @staticmethod
    def _approx_quadratic(
            x: torch.Tensor,   # (T+1, x_dims+u_dims) or (T+1, b, xu)
            u: torch.Tensor,   # (T+1, x_dims+u_dims) or (T+1, b, xu)
            theta: torch.Tensor,
            gt_neighbors_batch: torch.Tensor,  # (N-1 or 0, T, K, 2) or list of same for batch
            mus_batch: torch.Tensor,  # (1 or N, T, K, 2) or list of same for batch
            probs_batch: torch.Tensor,  # (1 or N, T, K)  or list of same for batch
            goal: torch.Tensor,  # (2, ) or (b, 2)
            lanes: torch.Tensor,
            rbf_scale=2.0,
            diff=True):  #(T+1, 3, ) or (T+1, b, 3)
        """Directly computes a quadratic approximation of the cost."""

        # Deal with different time resolutions for prediction and planning
        predh = 6
        planh = x.shape[0]-1
        if planh != predh:
            num_repeat = planh // predh
            gt_neighbors_batch = None if gt_neighbors_batch is None else torch.repeat_interleave(gt_neighbors_batch, num_repeat, dim=1)
            mus_batch = None if mus_batch is None else torch.repeat_interleave(mus_batch, num_repeat, dim=1)
            lanes = torch.repeat_interleave(lanes, num_repeat, dim=0)[num_repeat-1:]

        # We are looking for a cost in the form:
        # cost = 1/2 * x^T*C*x + c^T*x 
        x = x.detach()
        u = u.detach()

        # the pad function is in inverse order dimensions 
        #   (1, 2) will pad the last dimension with 1 and 2
        #   (1, 2, 3, 4) will also pad the second to last dimension with 3 and 4
        pad = torch.nn.functional.pad  

        # ego_lane_lat = torch.square(x[..., :2] - lanes[..., :2]).sum(dim=-1)
        # ego_lane_heading = torch.square(x[..., 2] - lanes[..., 2])
        # We need to wrap the lane heading lanes[..., 2] with 2pi such that it is close to x[..., 2]. 
        lane_h = x[..., 2] - angle_wrap(x[..., 2] - lanes[..., 2])
        lanes = torch.stack((lanes[..., 0], lanes[..., 1], lane_h), dim=-1)
        # Now we can treat heading error term as if it was simply ego_lane_heading = torch.square(x[..., 2] - lanes[..., 2])
        zero_tensor = torch.zeros_like(theta[0])
        ego_lane_C = torch.diag(torch.stack((theta[0], theta[0], theta[1], zero_tensor))).unsqueeze(0).unsqueeze(1).type_as(x)  # ..., 4, 4
        ego_lane_c = -2. * lanes[..., :3] * torch.stack((theta[0], theta[0], theta[1])).unsqueeze(0).type_as(x)   # ..., 3
        ego_lane_c = torch.cat((ego_lane_c, torch.zeros_like(ego_lane_c)[..., :1]), dim=-1)  # ..., 4

        # goal only for last state
        # ego_goal = torch.square(x[-1, ..., :2] - goal).sum(dim=-1)
        ego_goal_C = torch.diag(torch.stack((theta[2], theta[2], zero_tensor, zero_tensor))).unsqueeze(0).unsqueeze(1).type_as(x)   # 1, 1, 4, 4
        ego_goal_c = -2. * goal.unsqueeze(0) * torch.stack((theta[2], theta[2])).unsqueeze(0).type_as(x)    # 1, 1, 2
        # pad along theta_dim and along time, so we only have nonzero only for last step
        ego_goal_C = pad(ego_goal_C, (0, 0, 0, 0, 0, 0, x.shape[0]-1, 0))  # T, 1, 4, 4
        ego_goal_c = pad(ego_goal_c, (0, 2, 0, 0, x.shape[0]-1, 0))  # T, 1, 4

        control_C = torch.diag(torch.stack((theta[3], theta[3]))).unsqueeze(0).unsqueeze(1).type_as(x)   # ..., 2, 2

        _, grads, hessians = LinearCost1._collision_reward(
                mus_batch, probs_batch, x, gt_neighbors=gt_neighbors_batch, rbf_scale=rbf_scale, return_grad=True)
        grads = -grads * theta[4]
        hessians = -hessians * theta[4]

        # hessian matrix * tau vector. Using matmul to do this for last two dims, keeping T, batch
        # This comes from the Taylor approximation, expanding the quadratic term with Hessian will give a linear term.
        grads = grads - torch.matmul(hessians, x[..., :2].unsqueeze(-1)).squeeze(-1)            
        if not diff:
            hessians = hessians.detach()
            grads = grads.detach()

        # Combine elements
        state_C = 2. * (ego_lane_C + ego_goal_C)  # t, 1, 4, 4
        state_c = ego_lane_c + ego_goal_c  # t, b, 4
        
        C = state_C
        C = torch.nn.functional.pad(C, (0, 2, 0, 2))  # t, 1, 6, 6
        C = C.repeat(1, hessians.shape[1], 1, 1)  # t, b, 6, 6
        C[..., :2, :2] += hessians
        C[..., -2:, -2:] += 2. * control_C 
        c = torch.nn.functional.pad(state_c, (0, 2))  # t, b, 6
        c[..., :2] += grads

        return C, c


class LinearCostAngleBug(LinearCost1):

    @staticmethod
    def _compute_cost_terms(
            xu: torch.Tensor,   # (T+1, x_dims+u_dims) or (T+1, b, xu)
            gt_neighbors: torch.Tensor,  # (N-1 or 0, T, K, 2) or list of same for batch
            mus: torch.Tensor,  # (1 or N, T, K, 2) or list of same for batch
            probs: torch.Tensor,  # (1 or N, T, K)  or list of same for batch
            goal: torch.Tensor,  # (2, ) or (b, 2)
            lanes: torch.Tensor,
            rbf_scale: float,
            ):  #(T+1, 3, ) or (T+1, b, 3)

        x, u = torch.split(xu, (4, 2), dim=-1)  # x, y, orient, vel, d_orient, acc

        # Deal with different time resolutions for prediction and planning
        predh = lanes.shape[0]-1
        planh = x.shape[0]-1
        if planh != predh:
            assert planh % predh == 0 and planh > predh
            num_repeat = planh // predh
            gt_neighbors = None if gt_neighbors is None else torch.repeat_interleave(gt_neighbors, num_repeat, dim=1)
            mus = None if mus is None else torch.repeat_interleave(mus, num_repeat, dim=1)
            lanes = torch.repeat_interleave(lanes, num_repeat, dim=0)[num_repeat-1:]

        assert lanes.shape[:-1] == x.shape[:-1]
        assert len(goal.shape) == len(x.shape)-1

        ego_lane_lat = torch.square(x[..., :2] - lanes[..., :2]).sum(dim=-1)
        # ego_lane_heading = torch.square(angle_wrap(x[..., 2] - lanes[..., 2]))
        ego_lane_heading = torch.square(x[..., 2] - lanes[..., 2])
        ego_goal = torch.cat((torch.zeros_like(ego_lane_heading)[:-1],
                              torch.square(x[-1, ..., :2] - goal).sum(dim=-1).unsqueeze(0)), dim=0) * 0.1
        control_cost = torch.square(u).sum(dim=-1)

        collision_reward = LinearCost1._collision_reward(
            mus, probs, x, gt_neighbors=gt_neighbors, rbf_scale=rbf_scale)

        cost_terms = [
            ego_lane_lat, 
            ego_lane_heading,
            ego_goal,
            control_cost,
            -collision_reward]

        cost_terms = torch.stack(cost_terms, dim=-1)  # t, (b), theta_dim            
        return cost_terms


    @staticmethod
    def _approx_quadratic(
            x: torch.Tensor,   # (T+1, x_dims+u_dims) or (T+1, b, xu)
            u: torch.Tensor,   # (T+1, x_dims+u_dims) or (T+1, b, xu)
            theta: torch.Tensor,
            gt_neighbors_batch: torch.Tensor,  # (N-1 or 0, T, K, 2) or list of same for batch
            mus_batch: torch.Tensor,  # (1 or N, T, K, 2) or list of same for batch
            probs_batch: torch.Tensor,  # (1 or N, T, K)  or list of same for batch
            goal: torch.Tensor,  # (2, ) or (b, 2)
            lanes: torch.Tensor,
            rbf_scale=2.0,
            diff=True):  #(T+1, 3, ) or (T+1, b, 3)
        """Directly computes a quadratic approximation of the cost."""

        # Deal with different time resolutions for prediction and planning
        predh = 6
        planh = x.shape[0]-1
        if planh != predh:
            num_repeat = planh // predh
            gt_neighbors_batch = None if gt_neighbors_batch is None else torch.repeat_interleave(gt_neighbors_batch, num_repeat, dim=1)
            mus_batch = None if mus_batch is None else torch.repeat_interleave(mus_batch, num_repeat, dim=1)
            lanes = torch.repeat_interleave(lanes, num_repeat, dim=0)[num_repeat-1:]

        # We are looking for a cost in the form:
        # cost = 1/2 * x^T*C*x + c^T*x 
        x = x.detach()
        u = u.detach()

        # the pad function is in inverse order dimensions 
        #   (1, 2) will pad the last dimension with 1 and 2
        #   (1, 2, 3, 4) will also pad the second to last dimension with 3 and 4
        pad = torch.nn.functional.pad  

        # ego_lane_lat = torch.square(x[..., :2] - lanes[..., :2]).sum(dim=-1)
        # ego_lane_heading = torch.square(x[..., 2] - lanes[..., 2])
        # # We need to wrap the lane heading lanes[..., 2] with 2pi such that it is close to x[..., 2]. 
        # lane_h = x[..., 2] - angle_wrap(x[..., 2] - lanes[..., 2])
        # lanes = torch.stack((lanes[..., 0], lanes[..., 1], lane_h), dim=-1)
        # Now we can treat heading error term as if it was simply ego_lane_heading = torch.square(x[..., 2] - lanes[..., 2])
        zero_tensor = torch.zeros_like(theta[0])
        ego_lane_C = torch.diag(torch.stack((theta[0], theta[0], theta[1], zero_tensor))).unsqueeze(0).unsqueeze(1).type_as(x)  # ..., 4, 4
        ego_lane_c = -2. * lanes[..., :3] * torch.stack((theta[0], theta[0], theta[1])).unsqueeze(0).type_as(x)   # ..., 3
        ego_lane_c = torch.cat((ego_lane_c, torch.zeros_like(ego_lane_c)[..., :1]), dim=-1)  # ..., 4

        # goal only for last state
        # ego_goal = torch.square(x[-1, ..., :2] - goal).sum(dim=-1)
        ego_goal_C = torch.diag(torch.stack((theta[2], theta[2], zero_tensor, zero_tensor))).unsqueeze(0).unsqueeze(1).type_as(x)   # 1, 1, 4, 4
        ego_goal_c = -2. * goal.unsqueeze(0) * torch.stack((theta[2], theta[2])).unsqueeze(0).type_as(x)    # 1, 1, 2
        # pad along theta_dim and along time, so we only have nonzero only for last step
        ego_goal_C = pad(ego_goal_C, (0, 0, 0, 0, 0, 0, x.shape[0]-1, 0))  # T, 1, 4, 4
        ego_goal_c = pad(ego_goal_c, (0, 2, 0, 0, x.shape[0]-1, 0))  # T, 1, 4

        control_C = torch.diag(torch.stack((theta[3], theta[3]))).unsqueeze(0).unsqueeze(1).type_as(x)   # ..., 2, 2

        _, grads, hessians = LinearCost1._collision_reward(
                mus_batch, probs_batch, x, gt_neighbors=gt_neighbors_batch, rbf_scale=rbf_scale, return_grad=True)
        grads = -grads * theta[4]
        hessians = -hessians * theta[4]

        # hessian matrix * tau vector. Using matmul to do this for last two dims, keeping T, batch
        # This comes from the Taylor approximation, expanding the quadratic term with Hessian will give a linear term.
        grads = grads - torch.matmul(hessians, x[..., :2].unsqueeze(-1)).squeeze(-1)            
        if not diff:
            hessians = hessians.detach()
            grads = grads.detach()

        # Combine elements
        state_C = 2. * (ego_lane_C + ego_goal_C)  # t, 1, 4, 4
        state_c = ego_lane_c + ego_goal_c  # t, b, 4
        
        C = state_C
        C = torch.nn.functional.pad(C, (0, 2, 0, 2))  # t, 1, 6, 6
        C = C.repeat(1, hessians.shape[1], 1, 1)  # t, b, 6, 6
        C[..., :2, :2] += hessians
        C[..., -2:, -2:] += 2. * control_C 
        c = torch.nn.functional.pad(state_c, (0, 2))  # t, b, 6
        c[..., :2] += grads

        return C, c

class InterpretableLinearCost1(LinearCost1):
    """Interpretable counterpart of LinearCost1. 

    This class computes more interpretable cost terms, and effectively assumes theta equals one. 
    The intended usage is to call forward with keep_components=True, which will return N interpretable cost terms.

    # TODO Add progress term instead of goal lateral distance
    # TODO Remove hardcoded dt=0.5 and predh=6
    """
    theta_dim = 8

    def forward(self, xu, cost_inputs, keep_components=False):
        # return torch.zeros([xu.shape[0], xu.shape[1]], device=xu.device)
        gt_neighbors_batch, mus_batch, probs_batch, goal_batch, lanes, _ = cost_inputs
        dummy_theta = torch.ones_like(self.theta)
        return self._compute_cost(xu, dummy_theta, gt_neighbors_batch, mus_batch, probs_batch, goal_batch, lanes, rbf_scale=self.rbf_scale_long, keep_components=keep_components)

    @staticmethod
    def _compute_cost_terms(
            xu: torch.Tensor,   # (T+1, x_dims+u_dims) or (T+1, b, xu)
            gt_neighbors: torch.Tensor,  # (N-1 or 0, T, K, 2) or list of same for batch
            mus: torch.Tensor,  # (1 or N, T, K, 2) or list of same for batch
            probs: torch.Tensor,  # (1 or N, T, K)  or list of same for batch
            goal: torch.Tensor,  # (2, ) or (b, 2)
            lanes: torch.Tensor,
            rbf_scale: float,
            ):  #(T+1, 3, ) or (T+1, b, 3)

        x, u = torch.split(xu, (4, 2), dim=-1)  # x, y, orient, vel, d_orient, acc

        # Deal with different time resolutions for prediction and planning
        predh = lanes.shape[0]-1
        planh = x.shape[0]-1
        if planh != predh:
            num_repeat = planh // predh
            gt_neighbors = None if gt_neighbors is None else torch.repeat_interleave(gt_neighbors, num_repeat, dim=1)
            mus = None if mus is None else torch.repeat_interleave(mus, num_repeat, dim=1)
            lanes = torch.repeat_interleave(lanes, num_repeat, dim=0)[num_repeat-1:]

        assert lanes.shape[:-1] == x.shape[:-1]
        assert len(goal.shape) == len(x.shape)-1

        ego_lane_lat = torch.sqrt(torch.square(x[..., :2] - lanes[..., :2]).sum(dim=-1))
        ego_lane_heading =  torch.sqrt(torch.square(x[..., 2] - lanes[..., 2]))
        ego_goal = torch.repeat_interleave(
            torch.sqrt(torch.square(x[-1, ..., :2] - goal).sum(dim=-1)).unsqueeze(0), ego_lane_heading.shape[0],  dim=0)
        control_cost = torch.sqrt(torch.square(u).sum(dim=-1))
        control_cost1 = torch.sqrt(torch.square(u[..., 0]))
        control_cost2 = torch.sqrt(torch.square(u[..., 1]))

        prediction_reward = InterpretableLinearCost1._collision_reward(
            mus, probs, x, gt_neighbors=gt_neighbors, rbf_scale=rbf_scale)
        distance = InterpretableLinearCost1._distance_to_closest(
            mus, probs, x, gt_neighbors=gt_neighbors, rbf_scale=rbf_scale)

        cost_terms = [    
            ego_lane_lat,
            ego_lane_heading,
            ego_goal,
            control_cost,
            control_cost1,
            control_cost2,
            prediction_reward,
            distance]

        cost_terms = torch.stack(cost_terms, dim=-1)  # t, (b), theta_dim
        return cost_terms

    @staticmethod
    def _distance_to_closest(pred_mus: torch.Tensor, pred_probs: torch.Tensor, ego_x: torch.Tensor, gt_neighbors: torch.Tensor = None, rbf_scale = 2.0, return_grad: bool = False):
        """Distance to closest agent. For batched input we simply iterate over batch."""
        is_batched = (len(ego_x.shape) == 3)
        if is_batched:
            reward_outputs = []
            for b_i in range(ego_x.shape[1]):
                reward_outputs.append(InterpretableLinearCost1._distance_to_closest_single(  # recursive call to itslef
                    pred_mus[:, :, b_i], pred_probs[:, b_i], ego_x[:, b_i], 
                    gt_neighbors=(None if gt_neighbors is None else gt_neighbors[:, :, b_i]),
                    rbf_scale=rbf_scale,
                    return_grad=return_grad))

            if return_grad:
                # TODO would be nicer with zip
                prediction_reward = torch.stack([output[0] for output in reward_outputs], dim=1)
                gradients = torch.stack([output[1] for output in reward_outputs], dim=1)
                hessians = torch.stack([output[2] for output in reward_outputs], dim=1)
                return prediction_reward, gradients, hessians                
            else:
                prediction_reward = torch.stack(reward_outputs, dim=1)
                return prediction_reward
        else:      
            return InterpretableLinearCost1._distance_to_closest_single(pred_mus, pred_probs, ego_x, gt_neighbors, rbf_scale=rbf_scale, return_grad=return_grad)

    @staticmethod
    def _distance_to_closest_single(pred_mus: torch.Tensor, pred_probs: torch.Tensor, x: torch.Tensor, gt_neighbors: torch.Tensor = None, rbf_scale = 2.0, return_grad: bool = False):
        """Distance to closest agent for single input."""
        assert pred_mus.shape[0] == 0

        # Add in gt
        if gt_neighbors is not None:
            x_gt_delta = x[1:gt_neighbors.shape[1]+1, :2].unsqueeze(0) - torch.nan_to_num(gt_neighbors, nan=1e6)
            gt_comp_dists = tracable_norm(x_gt_delta, dim=-1)  # Distance from predictions (nodes, predhorizon)
            expected_dist = gt_comp_dists

        # Instead of closest agent, add distance term for all agents
        expected_dist = expected_dist.unsqueeze(-1)  # (N, T, 1)
        closest_dist = expected_dist.min(dim=0).values.min(dim=0).values  # (1,) 
        
        ret = closest_dist.repeat_interleave(expected_dist.shape[1], dim=0)  # (T, )
        ret = torch.cat((torch.zeros((1, ), dtype=ret.dtype, device=ret.device), ret), dim=0) # extend to (T+1, )

        if return_grad:
            return ret, None, None

        return ret

