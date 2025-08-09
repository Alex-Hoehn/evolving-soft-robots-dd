# adapted from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion

import abc
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable, Tuple, Union
from .models.diffusion import SEDD
from .models.task import TaskOptimizer
from . import graph_lib, noise_lib
import evogym
from evogym import is_connected, has_actuator


def sanitize_tokens(t: torch.Tensor, max_valid_id: int = 4) -> torch.Tensor:
    out = t.clone()
    out[out > max_valid_id] = 0
    return out

def sample_categorical_with_temperature(logits, temperature = 1.3) -> torch.Tensor:
    if temperature != 1.0:
        logits = logits / temperature

    probs = torch.softmax(logits, -1)
    idx = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
    return idx.view(probs.shape[:-1])

def classifier_free_guidance(cond_logits, uncond_logits, scale):
    return uncond_logits + scale * (cond_logits - uncond_logits)

def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")


def improved_task_sampling(
        task_model,
        initial_tokens: torch.Tensor,
        score: torch.Tensor,
        task_graph,
        structure_shape: Tuple[int, int],
        steps: int = 32,
        start_sigma: float = 5.0,
        eps: float = 1e-5,
        temperature: float = 1.3,
        guidance_scale: float = 1.0,
        enforce_validity: bool = True,
        max_attempts: int = 100,
        device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    task_model.eval()
    improved_tokens = []

    for i in range(initial_tokens.shape[0]):
        token_i = initial_tokens[i:i+1]
        score_i = score[i:i+1] if score.dim() > 1 else score[i:i+1].unsqueeze(1)

        attempts = 0
        found_valid = False

        while not found_valid and attempts < max_attempts:
            with torch.no_grad():
                # Add noise to structure
                sigma_tensor = torch.full((1,), start_sigma, device=device)
                noised_tokens = task_graph.sample_transition(token_i, sigma_tensor[:, None])

                x = noised_tokens
                sigmas = torch.linspace(start_sigma, eps, steps, device=device)

                for sigma in sigmas:
                    sigma_t = sigma.expand(1)

                    # Classifier-free guidance
                    cond_logits = task_model(x, score_i, sigma_t)
                    if guidance_scale > 0:
                        uncond_logits = task_model(x, None, sigma_t)
                        cond_logits = classifier_free_guidance(
                            cond_logits, uncond_logits, guidance_scale
                        )

                    x = sample_categorical_with_temperature(cond_logits, temperature)

                # print(x)
                improved = sanitize_tokens(x, max_valid_id=4)
                # print(improved)

                if enforce_validity:
                    struct = improved[0].reshape(*structure_shape).cpu().numpy()
                    # print(struct)
                    if is_connected(struct) and has_actuator(struct):
                        improved_tokens.append(improved[0])
                        found_valid = True
                else:
                    improved_tokens.append(improved[0])
                    found_valid = True

            attempts += 1

        if not found_valid:
            improved_tokens.append(initial_tokens[i])

    return torch.stack(improved_tokens)


_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, curr_sigma)

        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return sample_categorical(probs)


class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, t):
        sigma = self.noise(t)[0]

        score = score_fn(x, sigma)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]

        #return probs.argmax(dim=-1)
        return sample_categorical(probs)


def get_score_fn(model, train=False, sampling=False):
    def model_fn(x, sigma):
        if train:
            model.train()
        else:
            model.eval()
        return model(x, sigma)

    def score_fn(x, sigma):
        sigma = sigma.reshape(-1)
        score = model_fn(x, sigma)

        if sampling:
            return score.exp()
        return score

    return score_fn


def get_pc_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model, score_fn_factory=None):
        if score_fn_factory is None:
            sampling_score_fn = get_score_fn(model, train=False, sampling=True)
        else:
            sampling_score_fn = score_fn_factory(model)

        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            x = predictor.update_fn(sampling_score_fn, x, t, dt)


        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t)

        return x

    return pc_sampler


def sample_robot_structures(
        diffusion_model,
        task_model: Optional = None,
        task_id: Optional[torch.Tensor] = None,
        score: Optional[torch.Tensor] = None,
        batch_size: int = 32,
        structure_shape: Tuple[int, int] = (5, 5),
        steps: int = 64,
        device: torch.device = torch.device('cpu'),
        predictor: str = "analytic",
        denoise: bool = True,
        eps: float = 1e-5,
        enforce_validity: bool = True,
        max_attempts: int = 100,
        proj_fun: Optional[Callable] = None,
        temperature: float = 1.3,
        guidance_scale: float = 1.0,
        task_steps: int = 32,
        task_start_sigma: float = 5.0,
        **kwargs
) -> Tuple[torch.Tensor, np.ndarray]:
    graph = graph_lib.get_graph(diffusion_model.config, device)
    noise = noise_lib.get_noise(diffusion_model.config).to(device)
    length = diffusion_model.length

    print("Phase 1: Base Model generation...")

    valid_tokens = []
    attempts = 0

    while len(valid_tokens) < batch_size and attempts < max_attempts:
        needed = batch_size - len(valid_tokens)
        current_batch_size = min(needed * 2, batch_size)
        batch_dims = (current_batch_size, length)

        diffusion_sampler = get_pc_sampler(
            graph=graph,
            noise=noise,
            batch_dims=batch_dims,
            predictor=predictor,
            steps=steps,
            denoise=denoise,
            eps=eps,
            device=device,
            proj_fun=proj_fun if proj_fun else lambda x: x
        )

        tokens = diffusion_sampler(diffusion_model)
        tokens = sanitize_tokens(tokens, max_valid_id=4)

        for token in tokens:
            if len(valid_tokens) >= batch_size:
                break

            if enforce_validity:
                struct = token.reshape(*structure_shape).cpu().numpy()
                if is_connected(struct) and has_actuator(struct):
                    valid_tokens.append(token)
            else:
                valid_tokens.append(token)

        attempts += 1

    if len(valid_tokens) == 0:
        print("No valid structures generated")
        return torch.zeros((0, length), dtype=torch.long, device=device), np.zeros((0, *structure_shape))

    valid_tokens = valid_tokens[:batch_size]
    diffusion_tokens = torch.stack(valid_tokens).to(device)
    print(f"Phase 1 complete: {len(valid_tokens)} valid structures")

    if task_model is not None and score is not None:
        print("Phase 2: Task Model refinement...")

        task_graph = graph_lib.get_graph(task_model.config, device)

        final_tokens = improved_task_sampling(
            task_model=task_model,
            initial_tokens=diffusion_tokens,
            score=score,
            task_graph=task_graph,
            structure_shape=structure_shape,
            steps=task_steps,
            start_sigma=task_start_sigma,
            eps=eps,
            temperature=temperature,
            guidance_scale=guidance_scale,
            enforce_validity=enforce_validity,
            max_attempts=max_attempts,
            device=device
        )

        # Calculate improvements
        changes = [(diffusion_tokens[i] != final_tokens[i]).sum().item()
                   for i in range(len(final_tokens))]
        total_changes = sum(changes)
        structures_changed = sum(1 for c in changes if c > 0)

        print(f"Phase 2 complete:")
        print(f"  Structures changed: {structures_changed}/{len(final_tokens)}")
        print(f"  Total token changes: {total_changes}")

    else:
        final_tokens = diffusion_tokens
        print("Phase 2 skipped: no Task Model provided")

    # Convert to structures
    structures = final_tokens.view(final_tokens.shape[0], *structure_shape).cpu().numpy()

    return final_tokens, structures
