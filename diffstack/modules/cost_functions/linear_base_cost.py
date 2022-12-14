import torch
from typing import Dict, Iterable, Optional, Union, Any, Tuple


class LinearBaseCost(torch.nn.Module):
    """Cost function that is a linear combination of N cost terms.
    """
    def __init__(self, 
            theta: torch.TensorType, 
            normalized: Optional[bool] = False, 
            scaler: Optional[bool] = False, 
            is_trainable: Optional[bool] = False, 
            trainable_dims: Optional[Iterable] = None, 
            rbf_scale_long: Optional[float] = 2.0, 
            rbf_scale_lat: Optional[float] = 2.0, 
            trainable_rbf_scaler: Optional[bool] = False,
            control_limits: Optional[tuple] = None):
        super().__init__()

        self.control_limits = control_limits

        # Rbf
        if trainable_rbf_scaler:
            self.rbf_scale_long = torch.nn.Parameter(torch.Tensor([rbf_scale_long]).to(device=theta.device).squeeze().detach(), requires_grad=is_trainable)
            self.rbf_scale_lat = torch.nn.Parameter(torch.Tensor([rbf_scale_lat]).to(device=theta.device).squeeze().detach(), requires_grad=is_trainable)
        else:
            self.rbf_scale_long = rbf_scale_long
            self.rbf_scale_lat = rbf_scale_lat

        # Theta
        assert len(theta) == self.theta_dim
        if trainable_dims is None:
            trainable_dims = [True] * self.theta_dim
        assert len(trainable_dims) == self.theta_dim
        self.normalized = normalized
        self.trainable_dims = trainable_dims

        if normalized:
            # Normalized parameterization, theta always sums to the initial theta sum.
            theta_sum = (theta * torch.Tensor(trainable_dims).float().to(device=theta.device)).sum()
            log_theta = torch.log(theta / theta_sum).detach()  # normalize to sum to one                       
            self.log_theta = []
            self.fixed_log_theta = []
            for i in range(log_theta.shape[0]):
                attr_name = f"log_theta_{i+1}"
                if trainable_dims[i]:
                    setattr(self, attr_name, torch.nn.Parameter(log_theta[i], requires_grad=is_trainable)) 
                    self.log_theta.append(getattr(self, attr_name))
                else:
                    setattr(self, attr_name, torch.nn.Parameter(log_theta[i], requires_grad=False)) 
                    self.fixed_log_theta.append(getattr(self, attr_name))
            self.theta_params = None
            if scaler:
                self.theta_scaler = torch.nn.Parameter(theta_sum.unsqueeze(0), requires_grad=is_trainable)
            else:
                self.theta_scaler = theta_sum.detach()
            
            assert (torch.isclose(theta, self.theta).all().detach())

        else:
            # Direct parameterization
            self.log_theta = None
            if trainable_dims is not None and not all(trainable_dims):
                raise NotImplementedError
            else:        
                self.theta_params = [torch.nn.Parameter(theta[i], requires_grad=is_trainable) for i in range(theta.shape[0])]
            self.theta_scaler = 1.
            assert not scaler

    def forward(self, xu, cost_inputs, keep_components=False):
        raise NotImplementedError

    def approx_quadratic(self, x, u, cost_inputs, diff=True):
        raise NotImplementedError

    @property
    def theta(self):
        if self.normalized:
            # We cannot have the softmax op created in the init. Accessing theta through
            # this getter function will create a new op every time.
            th_train = iter(torch.unbind(torch.softmax(torch.stack(self.log_theta, dim=0), dim=0), dim=0))
            th_fixed = iter(self.fixed_log_theta)
            th = []
            for is_dim_trainable in self.trainable_dims:
                if is_dim_trainable:
                    th.append(next(th_train))
                else:
                    th.append(torch.exp(next(th_fixed)))
            return torch.stack(th, dim=0) * self.theta_scaler
        else:
            assert self.trainable_dims is None or all(self.trainable_dims)
            return torch.stack(self.theta_params, dim=0)

    @property
    def theta_standardized(self):
        # Decouple theta normalized to sum to one, and an overal scaler
        with torch.no_grad():
            theta_sum = self.theta.sum()
            return torch.cat((self.theta / theta_sum, theta_sum.unsqueeze(0)))

    def get_params_log(self):
        theta = self.theta_standardized.detach()
        log_dict = {}
        for i in range(theta.shape[0]):
            log_dict[f"theta_{i+1}"] = theta[i].item()
        if isinstance(self.rbf_scale_long, torch.nn.Parameter):
            log_dict[f"theta_rbf_lat"] = self.rbf_scale_lat.item()
            log_dict[f"theta_rbf_long"] = self.rbf_scale_long.item()
        return log_dict

    def get_params_summary_str(self):
        s = f"Plan cost theta: {self.theta.detach().cpu().numpy()}"
        s += f"\nPlan cost theta: {self.theta_standardized.detach().cpu().numpy()}"
        if isinstance(self.rbf_scale_long, torch.nn.Parameter):
            s += f" RBF scaler long={str(self.rbf_scale_long.detach().cpu().numpy())} lat={str(self.rbf_scale_lat.detach().cpu().numpy())}"
        return s

    def approximate_quadratic_autodiff_naive(self, x, u, cost_inputs=None, diff=True):
        """Adopted from mpc.approximate_cost"""
        with torch.enable_grad():
            tau = torch.cat((x, u), dim=2).detach()
            tau = torch.autograd.Variable(tau, requires_grad=True)

            grads = torch.autograd.functional.jacobian(
                lambda tau: self.forward(tau, cost_inputs).sum(), tau, create_graph=True, vectorize=True)
            hessians = list()
            for v_i in range(tau.shape[2]):  # over state dimensions
                hessians.append(
                    torch.autograd.grad(grads[..., v_i].sum(), tau, retain_graph=True)[0]
                )
            hessians = torch.stack(hessians, dim=-1)  # 7, 209, 6, 6

            # hessian matrix * tau vector. Using matmul to do this for last two dims, keeping T, batch
            grads = grads - torch.matmul(hessians, tau.unsqueeze(-1)).squeeze(-1)            
            
            if not diff:
                return hessians.detach(), grads.detach()
            return hessians, grads

    def gt_neighbors_gradient(self, xu, cost_inputs, gt_neighbors, insert_function=None):
        # Gradient of cost function wrt. gt_neighbors (gt poses of other agents).
        # TODO we could make it more efficient by computing only the collision term of the cost, 
        #   the gradient of other terms (independent of gt_neighbors) are always zero
        
        if insert_function is None:
            # The default way to insert the agent_xy into cost inputs is to replace gt_neighbors_batch
            # cost_inputs = gt_neighbors_batch, mus_batch, probs_batch, goal_batch, lanes, _ 
            insert_function = lambda cost_inputs, agent_xy: ([agent_xy] + list(cost_inputs[1:]))

        with torch.enable_grad():
            tau = torch.autograd.Variable(gt_neighbors, requires_grad=True)

            grads = torch.autograd.functional.jacobian(
                lambda tau: self.forward(xu, insert_function(cost_inputs, tau)).sum(), tau, create_graph=True, vectorize=True)
            # grads should have the same shape as gt_neighbors

        return grads.detach()
