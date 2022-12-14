import torch
from torch.autograd import Variable

from .lqr_step_refactored import LQRStepClass, get_cost

from mpc import util
from mpc.mpc import GradMethods


class TrajCostMPC(torch.nn.Module):
    """A differentiable box-constrained iLQR solver.

    This code is an extension of the mpc module from
    https://github.com/locuslab/mpc.pytorch
    for costs defined over trajectories, and custom nonlinear dynamics.

    This provides a differentiable solver for the following box-constrained
    control problem with a quadratic cost (defined by C and c) and
    non-linear dynamics (defined by f):

        min_{tau={x,u}} sum_t 0.5 tau_t^T C_t tau_t + c_t^T tau_t
                        s.t. x_{t+1} = f(x_t, u_t)
                            x_0 = x_init
                            u_lower <= u <= u_upper

    This implements the Control-Limited Differential Dynamic Programming
    paper with a first-order approximation to the non-linear dynamics:
    https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf

    Some of the notation here is from Sergey Levine's notes:
    http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_8_model_based_planning.pdf

    Required Args:
        n_state, n_ctrl, T

    Optional Args:
        u_lower, u_upper: The lower- and upper-bounds on the controls.
            These can either be floats or shaped as [T, n_batch, n_ctrl]
        u_init: The initial control sequence, useful for warm-starting:
            [T, n_batch, n_ctrl]
        lqr_iter: The number of LQR iterations to perform.
        grad_method: The method to compute the Jacobian of the dynamics.
            GradMethods.ANALYTIC: Use a manually-defined Jacobian.
                + Fast and accurate, use this if possible
            GradMethods.AUTO_DIFF: Use PyTorch's autograd.
                + Slow
            GradMethods.FINITE_DIFF: Use naive finite differences
                + Inaccurate
        delta_u (float): The amount each component of the controls
            is allowed to change in each LQR iteration.
        verbose (int):
            -1: No output or warnings
             0: Warnings
            1+: Detailed iteration info
        eps: Termination threshold, on the norm of the full control
             step (without line search)
        back_eps: `eps` value to use in the backwards pass.
        n_batch: May be necessary for now if it can't be inferred.
                 TODO: Infer, potentially remove this.
        linesearch_decay (float): Multiplicative decay factor for the
            line search.
        max_linesearch_iter (int): Can be used to disable the line search
            if 1 is used for some problems the line search can
            be harmful.
        exit_unconverged: Assert False if a fixed point is not reached.
        detach_unconverged: Detach examples from the graph that do
            not hit a fixed point so they are not differentiated through.
        backprop: Allow the solver to be differentiated through.
        slew_rate_penalty (float): Penalty term applied to
            ||u_t - u_{t+1}||_2^2 in the objective.
        prev_ctrl: The previous nominal control sequence to initialize
            the solver with.
        not_improved_lim: The number of iterations to allow that don't
            improve the objective before returning early.
        best_cost_eps: Absolute threshold for the best cost
            to be updated.
    """

    def __init__(
            self, n_state, n_ctrl, T,
            u_lower=None, u_upper=None,
            u_zero_I=None,
            u_init=None,
            lqr_iter=10,
            grad_method=GradMethods.ANALYTIC,
            delta_u=None,
            verbose=0,
            eps=1e-7,
            back_eps=1e-7,
            n_batch=None,
            linesearch_decay=0.2,
            max_linesearch_iter=10,
            exit_unconverged=True,
            detach_unconverged=True,
            backprop=True,
            slew_rate_penalty=None,
            prev_ctrl=None,
            not_improved_lim=5,
            best_cost_eps=1e-4
    ):
        super().__init__()

        assert (u_lower is None) == (u_upper is None)
        assert max_linesearch_iter > 0

        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.T = T
        self.u_lower = u_lower
        self.u_upper = u_upper

        if not isinstance(u_lower, float):
            self.u_lower = util.detach_maybe(self.u_lower)

        if not isinstance(u_upper, float):
            self.u_upper = util.detach_maybe(self.u_upper)

        self.u_zero_I = util.detach_maybe(u_zero_I)
        self.u_init = util.detach_maybe(u_init)
        self.lqr_iter = lqr_iter
        self.grad_method = grad_method
        self.delta_u = delta_u
        self.verbose = verbose
        self.eps = eps
        self.back_eps = back_eps
        self.n_batch = n_batch
        self.linesearch_decay = linesearch_decay
        self.max_linesearch_iter = max_linesearch_iter
        self.exit_unconverged = exit_unconverged
        self.detach_unconverged = detach_unconverged
        self.backprop = backprop
        self.not_improved_lim = not_improved_lim
        self.best_cost_eps = best_cost_eps

        self.slew_rate_penalty = slew_rate_penalty
        self.prev_ctrl = prev_ctrl

        self._lqr = LQRStepClass(
            n_state=self.n_state,
            n_ctrl=self.n_ctrl,
            T=self.T,
            u_lower=self.u_lower,
            u_upper=self.u_upper,
            u_zero_I=self.u_zero_I,
            delta_u=self.delta_u,
            linesearch_decay=self.linesearch_decay,
            max_linesearch_iter=self.max_linesearch_iter,
            delta_space=True,
            back_eps=self.back_eps,
        )

    # @profile
    def forward(self, x_init, cost, dx, cost_inputs, return_converged=False, return_iters=False):
        if self.n_batch is not None:
            n_batch = self.n_batch
        else:
            raise ValueError('MPC Error: Could not infer batch size, pass in as n_batch')

        assert x_init.ndimension() == 2 and x_init.size(0) == n_batch

        if self.u_init is None:
            u = torch.zeros(self.T, n_batch, self.n_ctrl).type_as(x_init.data)
        else:
            u = self.u_init
            if u.ndimension() == 2:
                u = u.unsqueeze(1).expand(self.T, n_batch, -1).clone()
        u = u.type_as(x_init.data)

        if self.verbose > 0:
            print('Initial mean(cost): {:.4e}'.format(
                torch.mean(get_cost(
                    self.T, u, cost, dx, x_init=x_init
                )).item()
            ))

        best = None

        # Add init trajectory
        if return_iters:
            iters = dict(x=[], u=[], cost=[])
            x = util.get_traj(self.T, u, x_init=x_init, dynamics=dx)
            iters['x'].append(x.detach())
            iters['u'].append(u.detach())
            iters['cost'].append(get_cost(self.T, u, cost, dx, x_init=x_init, cost_inputs=cost_inputs).detach())
        else:
            iters = dict()

        n_not_improved = 0
        for i in range(self.lqr_iter):
            u = Variable(util.detach_maybe(u), requires_grad=True)
            # Linearize the dynamics around the current trajectory.
            x = util.get_traj(self.T, u, x_init=x_init, dynamics=dx)
            F, f = dx.linearized(x, util.detach_maybe(u), diff=False)
            C, c = cost.approx_quadratic(x, u, cost_inputs, diff=False)

            x, u, back_out, for_out = self.solve_lqr_subproblem(
                cost_inputs, x_init, C, c, F, f, cost, dx, x, u)
            # back_out, for_out = _lqr.back_out, _lqr.for_out
            n_not_improved += 1

            assert x.ndimension() == 3
            assert u.ndimension() == 3

            # Add init trajectory
            if return_iters:
                iters['x'].append(x.detach())
                iters['u'].append(u.detach())
                iters['cost'].append(for_out.costs.detach())

            # Not improved means nothing in the batch is improved.
            if best is None:
                best = {
                    'x': list(torch.split(x, split_size_or_sections=1, dim=1)),
                    'u': list(torch.split(u, split_size_or_sections=1, dim=1)),
                    'costs': for_out.costs,
                    'full_du_norm': for_out.full_du_norm,
                    'alpha_du_norm': for_out.alpha_du_norm,
                }
                # TODO pkarkus this should set n_not_improved=1
            else:
                for j in range(n_batch):
                    # if for_out.costs[j] <= best['costs'][j] + self.best_cost_eps:
                    if for_out.costs[j] <= best['costs'][j] - self.best_cost_eps:
                        n_not_improved = 0
                    if for_out.costs[j] <= best['costs'][j]:
                        best['x'][j] = x[:,j].unsqueeze(1)
                        best['u'][j] = u[:,j].unsqueeze(1)
                        best['costs'][j] = for_out.costs[j]
                        best['full_du_norm'][j] = for_out.full_du_norm[j]
                        best['alpha_du_norm'][j] = for_out.alpha_du_norm[j]

            if self.verbose > 0:
                util.table_log('lqr', (
                    ('iter', i),
                    ('mean(cost)', torch.mean(best['costs']).item(), '{:.4e}'),
                    ('||full_du||_max', max(for_out.full_du_norm).item(), '{:.2e}'),
                    # ('||alpha_du||_max', max(for_out.alpha_du_norm), '{:.2e}'),
                    # TODO: alphas, total_qp_iters here is for the current
                    # iterate, not the best
                    ('mean(alphas)', for_out.mean_alphas.item(), '{:.2e}'),
                    ('total_qp_iters', back_out.n_total_qp_iter),
                ))

            # eps defines a max change of control to continue. We use the largest change over the batch.
            if max(for_out.full_du_norm) < self.eps or \
               n_not_improved > self.not_improved_lim:
                break


        x = torch.cat(best['x'], dim=1)
        u = torch.cat(best['u'], dim=1)
        full_du_norm = best['full_du_norm']
        alpha_du_norm = best['alpha_du_norm']

        F, f = dx.linearized(x, u, diff=True)            

        C, c = cost.approx_quadratic(x, u, cost_inputs, diff=True)

        # Run LQR without updating x and u
        x, u, _, _ = self.solve_lqr_subproblem(
            cost_inputs, x_init, C, c, F, f, cost, dx, x, u, no_op_forward=True)

        converged_mask = full_du_norm < self.eps
        # converged_mask = alpha_du_norm < self.eps
        if self.detach_unconverged:
            if not converged_mask.all():
                if self.exit_unconverged:
                    assert False

                if self.verbose >= 0:
                    print("LQR Warning: All examples did not converge to a fixed point.")
                    print("Detaching and *not* backpropping through the bad examples.")

                x = self.detach_unconverged_tensor(x, converged_mask)
                u = self.detach_unconverged_tensor(u, converged_mask)

        costs = best['costs']
        if return_converged:
            return (x, u, costs, converged_mask.detach(), iters)
        else:
            return (x, u, costs)

    @staticmethod
    def detach_unconverged_tensor(t, converged_mask):
        It = Variable(converged_mask.unsqueeze(0).unsqueeze(2)).type_as(t.data)
        t = t*It + t.clone().detach()*(1.-It)
        return t

    def solve_lqr_subproblem(self, cost_inputs, x_init, C, c, F, f, cost, dynamics, x, u,
                             no_op_forward=False):
        assert self.slew_rate_penalty is None or isinstance(cost, torch.nn.Module)
        e = Variable(torch.Tensor())
        x, u, back_out, for_out = self._lqr(x, u, cost, dynamics, no_op_forward, x_init, C, c, F, f if f is not None else e, cost_inputs)

        return x, u, back_out, for_out

