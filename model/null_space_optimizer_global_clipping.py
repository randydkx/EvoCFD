import math
from collections import defaultdict
import torch
from torch.optim.optimizer import Optimizer
import numpy as np


class Adam(Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ

    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, svd=False, thres=1.001,
                 weight_decay=0, amsgrad=False, apply_max_sv_clip=True, clipping_ratio_alpha = 0.01):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, svd=svd,
                        thres=thres)
        super(Adam, self).__init__(params, defaults)
        self.eigens = defaultdict(dict)
        self.transforms = defaultdict(dict)
        self.logger = None
        self.moment_MSV = {}
        self.tracking_alpha = 0.01
        self.apply_max_sv_clip = apply_max_sv_clip
        
        self.upperbound_stats = None
        
        self.show_result = False
        
        # --- per-parameter std smoothing states ---
        self.std_alpha = getattr(self, 'tracking_alpha', 0.2)
        self.std_eps = 1e-12
        self._sigma_prev = {}
        self._var_ewma = {}
        
        self.pid2name = {}
        
        self.NSP_module_name = None
        self._ptr_nsp_key = 0
        
        self.update_interval = 1

        # --- New states for global smoothed clipping ratio ---
        self.global_clipping_ratio = 1.0
        
        # windows_size = 1/self.clipping_ratio_alpha
        self.clipping_ratio_alpha = clipping_ratio_alpha


        
    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('svd', False)
            
    
    @torch.no_grad()
    def _ensure_init_clip_state(self, p, init_value):
        pid = id(p)
        if pid not in self.moment_MSV:
            self.moment_MSV[pid] = float(init_value)

    @torch.no_grad()
    def _ensure_pid_name_by_seq(self, p):
        pid = id(p)
        if pid in self.pid2name:
            return self.pid2name[pid]
        if self._ptr_nsp_key >= len(self.NSP_module_name):
            raise IndexError(f"NSP_module_name exhausted at index {self._ptr_nsp_key} while mapping pid={pid}.")
        name = self.NSP_module_name[self._ptr_nsp_key]
        self._ptr_nsp_key += 1
        self.pid2name[pid] = name
        if self.logger:
            self.logger.debug(f'pid = {pid} is mapped to name = {name} ... ')
        return name


    @torch.no_grad()
    def _update_sv_moment(self, p, current_sigma):
        pid = id(p)
        prev = self.moment_MSV[pid]
        alpha = self.tracking_alpha
        new_m = (1.0 - alpha) * prev + alpha * float(current_sigma)
        self.moment_MSV[pid] = new_m
        return new_m
    
    def get_moment_by_p(self, p):
        return self.moment_MSV[id(p)]

    @torch.no_grad()
    def _max_sigma_power_iter(self, W, n_iter=2, eps=1e-12):
        Wd = W.to(torch.float64)
        m, n = Wd.shape
        v = torch.randn(n, device=Wd.device, dtype=Wd.dtype)
        v = v / (v.norm() + eps)
        for _ in range(n_iter):
            u = Wd @ v
            u = u / (u.norm() + eps)
            v = Wd.t() @ u
            v = v / (v.norm() + eps)
        sigma = (Wd @ v).norm()
        u = (Wd @ v) / (sigma + eps)
        return sigma, u, v, Wd

    @torch.no_grad()
    def _clip_update_only_max_sv_rank1_(self, update_2d, tau, n_iter=2, eps=1e-12, verify=False):
        sigma, u, v, Wd = self._max_sigma_power_iter(update_2d, n_iter=n_iter, eps=eps)
        sigma_before = float(sigma)
        if sigma > tau:
            alpha_scale = float(tau / (sigma + eps))
            Wv = (Wd @ v)
            Wd = Wd + (alpha_scale - 1.0) * (Wv[:, None] @ v[None, :])
            if verify:
                sigma_after = float(self._max_sigma_power_iter(Wd, n_iter=n_iter, eps=eps)[0])
            else:
                sigma_after = max(tau, 0.0)
        else:
            sigma_after = sigma_before
        update_clipped = Wd.to(update_2d.dtype)
        return sigma_before, sigma_after, update_clipped

    @torch.no_grad()
    def _update_param_std(self, p, current_sigma: float):
        pid = id(p)
        m = float(self.moment_MSV[pid])
        diff = float(current_sigma) - m
        sq = diff * diff
        if pid not in self._var_ewma:
            self._var_ewma[pid] = sq
        alpha = self.std_alpha
        v_prev = float(self._var_ewma[pid])
        v_new = (1.0 - alpha) * v_prev + alpha * sq
        self._var_ewma[pid] = v_new
        std_p = float(np.sqrt(max(v_new, self.std_eps)))
        return std_p

    def get_std_by_p(self, p):
        return float(np.sqrt(max(self._var_ewma[id(p)], self.std_eps)))

    @torch.no_grad()
    def _compute_tau_for_param(self, p, m: float, std_p: float):
        upperbound_1 = None
        name = self.pid2name.get(id(p), None)
        if isinstance(self.upperbound_stats, dict) and name is not None:
            entry = self.upperbound_stats.get(name, None)
            if isinstance(entry, dict) and 'upperbound_1' in entry:
                try:
                    upperbound_1 = float(entry['upperbound_1'])
                except Exception:
                    upperbound_1 = None
        if upperbound_1 is not None and not math.isnan(upperbound_1):
            tau = upperbound_1 if (m < upperbound_1) else (m + std_p)
        else:
            tau = m + std_p
        if tau is None or not np.isfinite(tau):
            tau = max(m, 0.0)
        else:
            tau = max(tau, 0.0)
        return float(tau)

    def satisfy(self, p):
        return True
        # sel_layer = ["layers.2.attention.W_q", "layers.2.attention.W_k",
        #              "layers.2.attention.W_v", "layers.2.attention.W_out",
        #              "layers.2.linear0", "layers.2.linear1"]
        # if self.pid2name.get(id(p), None) in sel_layer:
        #     return True
        # else:
        #     return False
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        _step = 0
        if self.param_groups and self.param_groups[0]['params']:
            first_param = self.param_groups[0]['params'][0]
            if first_param in self.state:
                _step = self.state[first_param].get('step', 0)

        clippable_updates = []
        individual_ratios = []
        maximum_singular_values = []
        tau_values = []

        for group in self.param_groups:
            svd = group['svd']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients')
                
                update = self.get_update(group, grad, p)
                p._optim_update_cache = update

                if svd and update.ndim == 2:
                    self._ensure_init_clip_state(p, 0)
                    _ = self._ensure_pid_name_by_seq(p)
                    
                    if (_step > 0 and (_step - 1) % self.update_interval == 0 and self.apply_max_sv_clip and self.satisfy(p)):
                         clippable_updates.append(p)
        
        if clippable_updates:
            for p in clippable_updates:
                update = p._optim_update_cache
                sigma_max, _, _, _ = self._max_sigma_power_iter(update, n_iter=10)
                sigma_max_val = float(sigma_max)
                
                m = self._update_sv_moment(p, sigma_max_val)
                std_p = self._update_param_std(p, sigma_max_val)
                tau = self._compute_tau_for_param(p, m, std_p)
                
                ratio = min(1.0, tau / (sigma_max_val + 1e-8))
                individual_ratios.append(ratio)
                
                maximum_singular_values.append(sigma_max_val)
                tau_values.append(tau)
            
            if individual_ratios:
                current_avg_ratio = sum(individual_ratios) / len(individual_ratios)
                self.global_clipping_ratio = (1.0 - self.clipping_ratio_alpha) * self.global_clipping_ratio + \
                                              self.clipping_ratio_alpha * current_avg_ratio

        for group in self.param_groups:
            svd = group['svd']
            for p in group['params']:
                if not hasattr(p, '_optim_update_cache'): continue
                
                update = p._optim_update_cache
                update_clipped = update

                if svd and p.ndim == 2 and (_step > 0 and (_step - 1) % self.update_interval == 0 and self.apply_max_sv_clip and self.satisfy(p)):
                    update_clipped = self.global_clipping_ratio * update
                
                if svd and len(self.transforms) > 0 and p in self.transforms:
                    update_ = torch.mm(update_clipped, self.transforms[p])
                else:
                    update_ = update_clipped
                
                p.data.add_(update_)
                del p._optim_update_cache

        # --- MODIFIED LOGGING SECTION ---
        if self.show_result and self.logger and (_step > 0 and (_step - 1) % 10 == 0):
            self.logger.debug(f'Step {_step}:')
            
            # Manually format the list output to avoid quotes
            sv_str = ', '.join([f'{v:.4f}' for v in maximum_singular_values])
            self.logger.debug(f'maximum singular values of updates: [{sv_str}]')
            
            if self.apply_max_sv_clip:
                tau_str = ', '.join([f'{v:.4f}' for v in tau_values])
                self.logger.debug(f'tau values used for clipping: [{tau_str}]')
                
                norm_clipping_ratios = [min(1.0, t / (s + 1e-8)) for t, s in zip(tau_values, maximum_singular_values)]
                ratios_str = ', '.join([f'{r:.4f}' for r in norm_clipping_ratios])
                self.logger.debug(f'norm clipping ratios: [{ratios_str}]')
                
                self.logger.debug(f'Applied global clipping ratio: {self.global_clipping_ratio:.4f}')

            
            svd_2d_params = [p for group in self.param_groups for p in group["params"] if p.ndim == 2 and group["svd"] and id(p) in self.moment_MSV]
            if svd_2d_params:
                moments_str = ', '.join([f'{self.moment_MSV[id(p)]:.4f}' for p in svd_2d_params])
                self.logger.debug(f'moment: [{moments_str}]')
            
            self.logger.debug(f'upperbound statistics: {self.upperbound_stats}')
        
        return loss

    def get_transforms(self):
        for group in self.param_groups:
            svd = group['svd']
            if svd is False:
                continue
            for p in group['params']:
                if p not in self.eigens:
                    continue
                thres = group['thres']
                num_eigen = group['num_eigen']
                # ind = self.eigens[p]['eigen_value'] <= self.eigens[p]['eigen_value'][-1] * thres
                ind = self.eigens[p]['eigen_value'] <= self.eigens[p]['eigen_value'][-num_eigen]
                # condition = sigma_max / sigma_min
                self.logger.debug('reserving basis {}/{}; cond: {}, radio:{}'.format(
                    ind.sum(), self.eigens[p]['eigen_value'].shape[0],
                    self.eigens[p]['eigen_value'][0] / self.eigens[p]['eigen_value'][-1],
                    self.eigens[p]['eigen_value'][ind].sum() / self.eigens[p]['eigen_value'].sum()
                ))
                # GVV^T
                # get the columns
                basis = self.eigens[p]['eigen_vector'][:, ind]
                transform = torch.mm(basis, basis.transpose(1, 0))
                self.transforms[p] = transform / torch.norm(transform)
                self.transforms[p].detach_()

    def get_eigens(self, fea_in):
        self.logger.debug(f'fea_in.keys().length = {len(fea_in.keys())} ... ')
        for group in self.param_groups:
            svd = group['svd']
            self.logger.debug('group_SVD = {}'.format(group['svd']))
            if svd is False:
                self.logger.debug('svd = false')
                continue
            for p in group['params']:
                if (p not in fea_in):
                    # self.logger.debug('p.grad is None or p not in fea_in')
                    continue
                cov = fea_in[p]
                if cov is None:
                    continue
                if not torch.isfinite(cov).all():
                    if self.logger:
                        self.logger.warning('Skipping SVD for parameter due to non-finite covariance values.')
                    continue
                if cov.abs().sum() == 0:
                    if self.logger:
                        self.logger.debug('Skipping SVD for parameter due to zero covariance matrix.')
                    continue
                eigen = self.eigens[p]
                self.logger.debug(f'eigen values calculation ... fea_in[p].shape = {cov.shape}')
                _, eigen_value, eigen_vector = torch.svd(cov, some=False)
                eigen['eigen_value'] = eigen_value
                eigen['eigen_vector'] = eigen_vector
    def get_update(self, group, grad, p):
        amsgrad = group['amsgrad']
        state = self.state[p]
        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p.data)
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = group['betas']
        state['step'] += 1
        if group['weight_decay'] != 0:
            grad.add_(group['weight_decay'], p.data)
        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = max_exp_avg_sq.sqrt().add_(group['eps'])
        else:
            denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
        update = - step_size * exp_avg / denom
        return update