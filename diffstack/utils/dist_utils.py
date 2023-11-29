import torch
from torch.nn.functional import one_hot, gumbel_softmax
import diffstack.utils.tensor_utils as TensorUtils
from diffstack.dynamics.base import Dynamics
from diffstack.dynamics.unicycle import Unicycle
import abc
import numpy as np
import diffstack.utils.geometry_utils as GeoUtils


def categorical_sample(pi, num_sample):
    bs, M = pi.shape
    pi0 = torch.cat([torch.zeros_like(pi[:, :1]), pi[:, :-1]], -1)
    pi_cumsum = pi0.cumsum(-1)
    rand_num = torch.rand([bs, num_sample], device=pi.device)
    flag = rand_num.unsqueeze(-1) > pi_cumsum.unsqueeze(-2)
    idx = torch.argmax((flag * torch.arange(0, M, device=pi.device)[None, None]), -1)
    return idx


def categorical_psample_wor(
    logpi, num_sample, num_factor=None, factor_mask=None, relevance_score=None
):
    """pseodo-sample (pick the most probable modes) from a categorical distribution without replacement
    the distribution is assumed to be a factorized categorical distribution

    Args:
        logpi (probability): [B,M,D], M is the number of factors, D is the number of categories
        num_sample (_type_): number of samples
        num_factor (int): If not None, pick the top num_factor factors with the highest probability variation

    """
    B, M, D = logpi.shape
    if num_factor is None:
        num_factor = M
    # assert D**num_factor>=num_sample
    if num_factor != M:
        if relevance_score is None:
            # default to use the maximum mode probability as measure
            relevance_score = -logpi.max(-1)[0]
        if factor_mask is not None:
            relevance_score.masked_fill_(
                torch.logical_not(factor_mask), relevance_score.min() - 1
            )
        idx = torch.topk(relevance_score, num_factor, dim=1)[1]
    else:
        idx = torch.arange(M, device=logpi.device)[None, :].repeat_interleave(B, 0)
    factor_logpi = torch.gather(
        logpi, 1, idx.unsqueeze(-1).repeat_interleave(D, -1)
    )  # Factors are chosen
    # calculate the product log probability of the factors
    num_factor = factor_logpi.shape[1]

    prod_logpi = sum(
        [
            factor_logpi[:, i].reshape(B, *[1] * i, D, *[1] * (num_factor - i - 1))
            for i in range(num_factor)
        ]
    )  # D**num_factor

    prod_logpi_flat = prod_logpi.view(B, -1)
    factor_samples = torch.topk(prod_logpi_flat, num_sample, 1)[1]
    factor_sample_idx = list()
    for i in range(num_factor):
        factor_sample_idx.append(factor_samples % D)
        factor_samples = torch.div(factor_samples, D, rounding_mode="floor")
    factor_sample_idx = torch.stack(factor_sample_idx, -1).flip(-1)
    # for unselected factors, pick the maximum likelihood mode
    samples = logpi.argmax(-1).unsqueeze(1).repeat_interleave(num_sample, 1)
    samples.scatter_(
        2, idx.unsqueeze(1).repeat_interleave(num_sample, 1), factor_sample_idx
    )

    return samples, idx

    # nonfactor_pi = torch.gather(pi,1,nonidx.unsqueeze(-1).repeat_interleave(D,-1))


class BaseDist(abc.ABC):
    @abc.abstractmethod
    def rsample(self, num_sample):
        pass

    @abc.abstractmethod
    def get_dist(self):
        pass

    @abc.abstractmethod
    def detach_(self):
        pass

    @abc.abstractmethod
    def index_select_(self, idx):
        pass


class MAGaussian(BaseDist):
    def __init__(self, mu, var, K, delta_x_clamp=1.0, min_var=1e-4):
        """multiagent gaussian distribution

        Args:
            mu (torch.Tensor): [B,N,D] mean
            var (torch.Tensor): [B,N,D] variance
            K (torch.Tensor): [B,N,D,L] coefficient of shared scene variance
            var = var + K @ K.T
        """
        self.mu = mu
        self.var = var
        self.K = K
        self.L = K.shape[-1]
        self.min_var = min_var
        self.delta_x_clamp = delta_x_clamp

    def rsample(self, num_sample):
        """sample from the distribution

        Args:
            sample_shape (torch.Size, optional): [description]. Defaults to torch.Size().

        Returns:
            torch.Tensor: [B,N,D] sample
        """
        B, N, D = self.mu.shape
        eps = torch.randn(B, num_sample, N, D, device=self.mu.device)
        scene_eps = torch.randn(B, num_sample, 1, self.L, device=self.mu.device)
        return (
            self.mu.unsqueeze(-3)
            + eps * self.var.sqrt().unsqueeze(-3)
            + (self.K.unsqueeze(1) @ scene_eps.unsqueeze(-1)).squeeze(-1)
        )

    def get_dist(self, mask):
        B, N, D = self.mu.shape

        var_diag = TensorUtils.block_diag_from_cat(
            torch.diag_embed(self.var + self.min_var)
        )
        var_inv_diag = torch.linalg.pinv(var_diag)
        C = torch.eye(self.K.shape[-1], device=self.K.device)[None].repeat_interleave(
            B, 0
        )
        K = self.K.reshape(B, N * D, self.L)
        K_T = K.transpose(-1, -2)
        var = var_diag + K @ K_T
        if self.L > N * D:
            var_inv = (
                var_inv_diag
                - var_inv_diag
                @ K
                @ torch.linalg.pinv(C + K_T @ var_inv_diag @ K)
                @ K_T
                @ var_inv_diag
            )
        else:
            var_inv = torch.linalg.pinv(var)
        return self.mu, var, var_inv

    def get_log_likelihood(self, xp, mask):
        # not tested
        B = self.mu.shape[0]
        mu, var, var_inv = self.get_dist(mask)
        if self.delta_x_clamp is not None:
            xp = torch.minimum(
                torch.maximum(xp, mu - self.delta_x_clamp), mu + self.delta_x_clamp
            )
        delta_x = xp - mu
        delta_x *= mask
        delta_x = delta_x.reshape(B, -1)
        mask_var = torch.diag_embed(
            (1 - mask).squeeze(-1).repeat_interleave(self.mu.shape[-1], -1)
        )
        log_prob = 0.5 * (
            torch.logdet(var_inv + mask_var)
            - (delta_x.unsqueeze(-2) @ var_inv @ delta_x.unsqueeze(-1)).flatten()
            + np.log(2 * np.pi)
        ).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        return log_prob

    def detach_(self):
        self.mu = self.mu.detach()
        self.var = self.var.detach()
        self.K = self.K.detach()

    def index_select_(self, idx):
        self.mu = self.mu[idx]
        self.var = self.varCategorical[idx]
        self.K = self.K[idx]


class MADynGaussian(BaseDist):
    def __init__(self, mu_u, var_u, var_x, K, dyn, delta_x_clamp=1.0, min_var=1e-4):
        """multiagent gaussian distribution with dynamics, input follows MAGaussian

        Args:
            mu_u (torch.Tensor): [B,N,udim] mean
            var_u (torch.Tensor): [B,N,udim] independent variance of u
            var_x (torch.Tensor): [B,N,xdim] independent variance of x
            K (torch.Tensor): [B,N,D,L] coefficient of shared scene variance
            dyn (Dynamics): dynamics object
            var = var + K @ K.T
        """
        self.mu_u = mu_u
        self.var_u = var_u
        self.var_x = var_x
        self.K = K
        self.L = K.shape[-1]
        self.dyn = dyn
        self.min_var = min_var
        self.delta_x_clamp = delta_x_clamp

    def rsample(self, x0, num_sample):
        """sample from the distribution

        Args:
            x0 (torch.Tensor): [B,N,xdim] initial state
            sample_shape (torch.Size, optional): [description]. Defaults to torch.Size().

        Returns:
            torch.Tensor: [B,N,xdim] sample
        """
        B, N, D = self.mu_u.shape
        eps = torch.randn(B, num_sample, N, D, device=self.mu_u.device)
        scene_eps = torch.randn(B, num_sample, 1, self.L, device=self.mu_u.device)
        u_sample = (
            self.mu_u.unsqueeze(1)
            + eps * self.var_u.sqrt().unsqueeze(1)
            + (self.K.unsqueeze(1) @ scene_eps.unsqueeze(-1)).squeeze(-1)
        )
        x_sample = self.dyn.step(
            x0.unsqueeze(1).repeat_interleave(num_sample, 1), u_sample
        )
        x_sample += self.var_x.sqrt().unsqueeze(1) * torch.randn_like(x_sample)
        return x_sample

    def get_dist(self, x0, mask):
        B, N, D = self.mu_u.shape

        mu_x, _, jacu = self.dyn.step(x0, self.mu_u, bound=False, return_jacobian=True)
        if mask is not None:
            jacu *= mask.unsqueeze(-1)
        # var_x_total = J@var_u@J.T+var_x
        var_x = jacu @ torch.diag_embed(self.var_u) @ jacu.transpose(
            -1, -2
        ) + torch.diag_embed(self.var_x + self.min_var)

        var_x_inv = torch.linalg.pinv(var_x)
        var_x_diag = TensorUtils.block_diag_from_cat(var_x)
        var_x_inv_diag = TensorUtils.block_diag_from_cat(var_x_inv)
        blk_jacu = TensorUtils.block_diag_from_cat(jacu)
        K = self.K.reshape(B, N * D, self.L)

        JK = blk_jacu @ K
        JK_T = JK.transpose(-1, -2)
        C = torch.eye(self.K.shape[-1], device=self.K.device)[None].repeat_interleave(
            B, 0
        )

        var = var_x_diag + JK @ JK_T

        if self.L > N * D:
            # Woodbury matrix identity
            var_inv = (
                var_x_inv_diag
                - var_x_inv_diag
                @ JK
                @ torch.linalg.pinv(C + JK_T @ var_x_inv_diag @ JK)
                @ JK_T
                @ var_x_inv_diag
            )
        else:
            var_inv = torch.linalg.pinv(var)
        return mu_x, var, var_inv

    def get_log_likelihood(self, x0, xp, mask):
        B, N = self.mu_u.shape[:2]
        mu_x, total_var_x, var_x_inv = self.get_dist(x0, mask)
        if self.delta_x_clamp is not None:
            xp = torch.minimum(
                torch.maximum(xp, mu_x - self.delta_x_clamp), mu_x + self.delta_x_clamp
            )
        # # ground truth
        delta_x = xp - mu_x
        delta_x *= mask
        mask_var = torch.diag_embed(
            (1 - mask).squeeze(-1).repeat_interleave(self.dyn.xdim, -1)
        )
        # hack: write delta function for each dynamics
        delta_x[..., 3] = GeoUtils.round_2pi(delta_x[..., 3])
        delta_x = delta_x.reshape(B, -1)
        log_prob = 0.5 * (
            torch.logdet(var_x_inv + mask_var)
            - (delta_x.unsqueeze(-2) @ var_x_inv @ delta_x.unsqueeze(-1)).flatten()
            + np.log(2 * np.pi)
        ).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        return log_prob

    def detach_(self):
        self.mu_u = self.mu_u.detach()
        self.var_u = self.var_u.detach()
        self.var_x = self.var_x.detach()
        self.K = self.K.detach()

    def index_select_(self, idx):
        self.mu_u = self.mu_u[idx]
        self.var_u = self.var_u[idx]
        self.var_x = self.var_x[idx]
        self.K = self.K[idx]


class MAGMM(BaseDist):
    def __init__(self, mu, var, K, pi, delta_x_clamp=1.0, min_var=1e-4):
        """multiagent gaussian distribution

        Args:
            mu (torch.Tensor): [B,M,N,D] mean
            var (torch.Tensor): [B,M,N,D] variance
            K (torch.Tensor): [B,M,N,D,L] coefficient of shared scene variance
            var = var + K @ K.T
            pi: [B,M] mixture weights
        """
        self.mu = mu
        self.var = var
        self.K = K
        self.L = K.shape[-1]
        self.M = mu.shape[1]
        self.pi = pi
        self.logits = torch.log(pi)
        self.pi_sum = pi.cumsum(-1)
        self.delta_x_clamp = delta_x_clamp
        self.min_var = min_var

    def rsample(self, num_sample, tau=None, infer=False):
        """sample from the distribution

        Args:
            sample_shape (torch.Size, optional): [description]. Defaults to torch.Size().

        Returns:
            torch.Tensor: [B,N,D] sample
        """
        B, M, N, D = self.mu.shape
        if tau is not None:
            mode = gumbel_softmax(
                self.logits[:, None].repeat_interleave(num_sample, 1), tau, hard=infer
            )
        else:
            mode = categorical_sample(self.pi, num_sample)
            mode = one_hot(mode, self.M)
        eps = torch.randn(B, M, num_sample, N, D, device=self.mu.device)
        scene_eps = torch.randn(B, M, num_sample, 1, self.L, device=self.mu.device)
        sample = (
            self.mu.unsqueeze(-3)
            + eps * self.var.sqrt().unsqueeze(-3)
            + (self.K.unsqueeze(2) @ scene_eps.unsqueeze(-1)).squeeze(-1)
        )
        sample = (sample * mode.transpose(1, 2).view(B, M, num_sample, 1, 1)).sum(
            1
        )  # B x num_sample x N x D
        return sample

    def get_dist(self, mask):
        B, M, N, D = self.mu.shape
        var_tiled, mu_tiled, K_tiled = TensorUtils.join_dimensions(
            (self.var, self.mu, self.K), 0, 2
        )
        var_diag = TensorUtils.block_diag_from_cat(
            torch.diag_embed(var_tiled + self.min_var)
        )
        var_inv_diag = torch.linalg.pinv(var_diag)
        C = torch.eye(K_tiled.shape[-1], device=self.K.device)[None].repeat_interleave(
            B * M, 0
        )
        K = K_tiled.reshape(B * M, N * D, self.L)
        K_T = K.transpose(-1, -2)
        var = var_diag + K @ K_T
        if self.L > N * D:
            var_inv = (
                var_inv_diag
                - var_inv_diag
                @ K
                @ torch.linalg.pinv(C + K_T @ var_inv_diag @ K)
                @ K_T
                @ var_inv_diag
            )
        else:
            var_inv = torch.linalg.pinv(var)
        return (
            self.mu,
            var.reshape(B, M, N * D, N * D),
            var_inv.reshape(B, M, N * D, N * D),
            self.pi,
        )

    def get_log_likelihood(self, xp, mask):
        # not tested
        B, M = self.mu.shape[:2]
        mu, var, var_inv, pi = self.get_dist(mask)
        xp = xp[:, None].repeat_interleave(M, 1)
        if self.delta_x_clamp is not None:
            xp = torch.minimum(
                torch.maximum(xp, mu - self.delta_x_clamp), mu + self.delta_x_clamp
            )
        delta_x = xp - mu

        delta_x *= mask.unsqueeze(1)
        delta_x = delta_x.reshape(B, M, -1)
        mask_var = (
            torch.diag_embed(
                (1 - mask).squeeze(-1).repeat_interleave(self.mu.shape[-1], -1)
            )
            .unsqueeze(1)
            .repeat_interleave(self.M, 1)
        )
        log_prob_mode = 0.5 * (
            torch.logdet(var_inv + mask_var)
            - (delta_x.unsqueeze(-2) @ var_inv @ delta_x.unsqueeze(-1)).reshape(B, M)
            + np.log(2 * np.pi)
        ).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        log_prob = torch.log((torch.exp(log_prob_mode) * pi).sum(-1))
        return log_prob

    def detach_(self):
        self.mu = self.mu.detach()
        self.var = self.var.detach()
        self.K = self.K.detach()
        self.pi = self.pi.detach()

    def index_select_(self, idx):
        self.mu = self.mu[idx]
        self.var = self.var[idx]
        self.K = self.K[idx]
        self.pi = self.pi[idx]


class MADynGMM(BaseDist):
    def __init__(self, mu_u, var_u, var_x, K, pi, dyn, delta_x_clamp=1.0, min_var=1e-6):
        """multiagent gaussian distribution with dynamics, input follows MAGaussian

        Args:
            mu_u (torch.Tensor): [B,M,N,udim] mean
            var_u (torch.Tensor): [B,M,N,udim] independent variance of u
            var_x (torch.Tensor): [B,M,N,xdim] independent variance of x
            K (torch.Tensor): [B,M,N,D,L] coefficient of shared scene variance
            pi (torch.Tensor): [B,M] mixture weights
            dyn (Dynamics): dynamics object
            var = var + K @ K.T
        """
        self.mu_u = mu_u
        self.var_u = var_u
        self.var_x = var_x
        self.K = K
        self.L = K.shape[-1]
        self.M = mu_u.shape[1]
        self.pi = pi
        self.logits = torch.log(pi)
        self.dyn = dyn
        self.delta_x_clamp = delta_x_clamp
        self.min_var = min_var

    def rsample(self, x0, num_sample, tau=None, infer=False):
        """sample from the distribution

        Args:
            sample_shape (torch.Size, optional): [description]. Defaults to torch.Size().

        Returns:
            torch.Tensor: [B,N,D] sample
        """
        B, M, N, udim = self.mu_u.shape
        if tau is not None:
            mode = gumbel_softmax(
                self.logits[:, None].repeat_interleave(num_sample, 1), tau, hard=infer
            )
        else:
            mode = categorical_sample(self.pi, num_sample)
            mode = one_hot(mode, self.M)
        eps = torch.randn(B, M, num_sample, N, udim, device=self.mu_u.device)
        scene_eps = torch.randn(B, M, num_sample, 1, self.L, device=self.mu_u.device)
        u_sample = (
            self.mu_u.unsqueeze(2)
            + eps * self.var_u.sqrt().unsqueeze(2)
            + (self.K.unsqueeze(2) @ scene_eps.unsqueeze(-1)).squeeze(-1)
        )
        x_sample = self.dyn.step(
            x0[:, None, None]
            .repeat_interleave(self.M, 1)
            .repeat_interleave(num_sample, 2),
            u_sample,
        )
        x_sample += self.var_x.sqrt().unsqueeze(2) * torch.randn_like(x_sample)
        x_sample = (x_sample * (mode.transpose(1, 2).view(B, M, num_sample, 1, 1))).sum(
            1
        )
        return x_sample

    def get_dist(self, x0, mask):
        B, M, N, udim = self.mu_u.shape
        var_x_tiled, var_u_tiled, mu_u_tiled, K_tiled = TensorUtils.join_dimensions(
            (self.var_x, self.var_u, self.mu_u, self.K), 0, 2
        )
        x0_tiled = x0.repeat_interleave(M, 0)
        mu_x, _, jacu = self.dyn.step(
            x0_tiled, mu_u_tiled, bound=False, return_jacobian=True
        )
        # var_x_total = J@var_u@J.T+var_x
        var_x = jacu @ torch.diag_embed(var_u_tiled) @ jacu.transpose(
            -1, -2
        ) + torch.diag_embed(var_x_tiled + self.min_var)

        var_x_inv = torch.linalg.pinv(var_x)
        var_x_diag = TensorUtils.block_diag_from_cat(var_x)
        var_x_inv_diag = TensorUtils.block_diag_from_cat(var_x_inv)
        blk_jacu = TensorUtils.block_diag_from_cat(jacu)
        K = K_tiled.reshape(B * M, N * udim, self.L)
        JK = blk_jacu @ K
        JK_T = JK.transpose(-1, -2)
        C = torch.eye(K_tiled.shape[-1], device=self.K.device)[None].repeat_interleave(
            B * M, 0
        )

        var = var_x_diag + JK @ JK_T
        if self.L > N * udim:
            # Woodbury matrix identity
            var_inv = (
                var_x_inv_diag
                - var_x_inv_diag
                @ JK
                @ torch.linalg.pinv(C + JK_T @ var_x_inv_diag @ JK)
                @ JK_T
                @ var_x_inv_diag
            )
        else:
            var_inv = torch.linalg.pinv(var)
        return (
            *TensorUtils.reshape_dimensions((mu_x, var, var_inv), 0, 1, (B, M)),
            self.pi,
        )

    def get_log_likelihood(self, x0, xp, mask):
        B, M, N = self.mu_u.shape[:3]
        mu_x, total_var_x, var_x_inv, pi = self.get_dist(x0, mask)

        xp = xp[:, None].repeat_interleave(M, 1)
        if self.delta_x_clamp is not None:
            xp = torch.minimum(
                torch.maximum(xp, mu_x - self.delta_x_clamp), mu_x + self.delta_x_clamp
            )
        # # ground truth
        delta_x = xp - mu_x

        delta_x *= mask.unsqueeze(1)
        delta_x = delta_x.reshape(B, M, -1)
        mask_var = (
            torch.diag_embed(
                (1 - mask).squeeze(-1).repeat_interleave(self.dyn.xdim, -1)
            )
            .unsqueeze(1)
            .repeat_interleave(self.M, 1)
        )
        log_prob_mode = 0.5 * (
            torch.logdet(var_x_inv + mask_var)
            - (delta_x.unsqueeze(-2) @ var_x_inv @ delta_x.unsqueeze(-1)).reshape(B, M)
            + np.log(2 * np.pi)
        ).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        # log_prob = torch.log((torch.exp(log_prob_mode)*pi).sum(-1))
        # Apply EM:
        log_prob = (log_prob_mode * pi).sum(-1)
        return log_prob

    def detach_(self):
        self.mu_u = self.mu_u.detach()
        self.var_u = self.var_u.detach()
        self.var_x = self.var_x.detach()
        self.K = self.K.detach()

    def index_select_(self, idx):
        self.mu_u = self.mu_u[idx]
        self.var_u = self.var_u[idx]
        self.var_x = self.var_x[idx]
        self.K = self.K[idx]


# class MADynEBM(BaseDist):
#     """Multiagent Energy-based model with dynamics.

#     """
#     def __init__(self,dyn)


def test_ma():
    bs = 5
    N = 3
    d = 2
    L = 10
    mu = torch.randn(bs, N, d)
    var = torch.randn(bs, N, d) ** 2
    K = torch.randn(bs, N, d, L)
    dist = MAGaussian(mu, var, K)
    sample = dist.rsample(15)
    var_inv = dist.get_dist()
    print("done")


def test_ma_dyn():
    bs = 5
    N = 3
    udim = 2
    xdim = 4
    L = 10
    dt = 0.1
    mu_u = torch.randn(bs, N, udim)
    var_u = torch.randn(bs, N, udim) ** 2
    var_x = torch.randn(bs, N, xdim) ** 2
    K = torch.randn(bs, N, udim, L)
    dyn = Unicycle(dt)
    dist = MADynGaussian(mu_u, var_u, var_x, K, dyn)
    x0 = torch.randn([bs, N, dyn.xdim])
    sample = dist.rsample(x0, 15)
    var_inv = dist.get_dist(x0)


def test_categorical():
    pi = torch.tensor([0.1, 0.3, 0.6])[None].repeat_interleave(6, 0)
    categorical_sample(pi, 10)


def test_GMM():
    D = 2
    B = 10
    M = 3
    N = 5
    L = 16

    mu = torch.randn([B, M, N, D])
    var = torch.randn([B, M, N, D]) ** 2
    K = torch.randn([B, M, N, D, L])
    pi = torch.rand([B, M])
    pi = pi / pi.sum(-1, keepdim=True)
    dist = MAGMM(mu, var, K, pi)
    sample = dist.rsample(10)
    _ = dist.get_dist()
    x = torch.randn([B, N, D])
    log_prob = dist.get_log_likelihood(x)


def test_dyn_GMM():
    udim = 2
    xdim = 4
    dt = 0.1
    B = 10
    M = 3
    N = 5
    L = 16
    dyn = Unicycle(dt)

    mu_u = torch.randn(B, M, N, udim)
    var_u = torch.randn(B, M, N, udim) ** 2
    var_x = torch.randn(B, M, N, xdim) ** 2
    K = torch.randn([B, M, N, udim, L])
    pi = torch.rand([B, M])
    pi = pi / pi.sum(-1, keepdim=True)
    dist = MADynGMM(mu_u, var_u, var_x, K, pi, dyn)

    x = torch.randn([B, N, xdim])
    xp = torch.randn([B, N, xdim])
    _ = dist.get_dist(x)
    sample = dist.rsample(x, 10, tau=0.5)
    log_prob = dist.get_log_likelihood(x, xp)
    print("done")


def test_sample_wor():
    pi = torch.randn([5, 6, 7])
    pi = pi**2
    pi = pi / pi.sum(-1, keepdim=True)
    sample = categorical_sample_wor(pi, 20, 3)


if __name__ == "__main__":
    test_sample_wor()
