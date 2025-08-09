# from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.

    Returns:
        A model function.
    """

    def model_fn(x, sigma):
        """Compute the output of the score-based model.

        Args:
            x: A mini-batch of input data.
            sigma: A mini-batch of noise levels.

        Returns:
            Model output
        """
        if train:
            model.train()
        else:
            model.eval()

        return model(x, sigma)

    return model_fn


def get_score_fn(model, train=False, sampling=False):
    """Get score function for SEDD models."""
    if sampling:
        assert not train, "Must sample in eval mode"
    model_fn = get_model_fn(model, train=train)

    def score_fn(x, sigma):
        sigma = sigma.reshape(-1)
        score = model_fn(x, sigma)

        if sampling:
            # when sampling return true score (not log used for training)
            return score.exp()

        return score

    return score_fn


def get_loss_fn(noise, graph, train, sampling_eps=1e-3, lv=False):
    """Get loss function for SEDD training."""

    def loss_fn(model, batch, cond=None, t=None, perturbed_batch=None):
        """
        Batch shape: [B, L] int. D given from graph
        """

        if t is None:
            if lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps

        sigma, dsigma = noise(t)

        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])

        log_score_fn = get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma)
        loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

        loss = (dsigma[:, None] * loss).sum(dim=-1)

        return loss

    return loss_fn


def get_optimizer(config, params):

    if hasattr(config, 'optim'):
        optim_config = config.optim
    else:
        # Default optimizer config
        optim_config = type('obj', (object,), {
            'optimizer': 'AdamW',
            'lr': 3e-4,
            'beta1': 0.9,
            'beta2': 0.999,
            'eps': 1e-8,
            'weight_decay': 0.0
        })

    if optim_config.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=optim_config.lr, betas=(optim_config.beta1, optim_config.beta2), eps=optim_config.eps,
                               weight_decay=optim_config.weight_decay)
    elif optim_config.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=optim_config.lr, betas=(optim_config.beta1, optim_config.beta2), eps=optim_config.eps,
                                weight_decay=optim_config.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {optim_config.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer,
                    scaler,
                    params,
                    step,
                    lr=3e-4,
                    warmup=1000,
                    grad_clip=1.0):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""

        if hasattr(config, 'optim'):
            lr = getattr(config.optim, 'lr', lr)
            warmup = getattr(config.optim, 'warmup', warmup)
            grad_clip = getattr(config.optim, 'grad_clip', grad_clip)

        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, graph, train, optimize_fn, accum):
    """Get step function for training/evaluation."""
    loss_fn = get_loss_fn(noise, graph, train)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, cond=None):
        nonlocal accum_iter
        nonlocal total_loss

        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = loss_fn(model, batch, cond=cond).mean() / accum

            if not torch.isfinite(loss):
                print("WARNING: infinite loss detected!", loss.item())
                return loss

            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])

                state['ema'].update(model.parameters())
                optimizer.zero_grad()

                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, cond=cond).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn