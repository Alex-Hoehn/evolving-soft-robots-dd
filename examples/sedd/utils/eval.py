import os
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

from evogym import get_full_connectivity
from stable_baselines3.common.env_util import make_vec_env

from ..models.diffusion import SEDD
from ..models.task import TaskOptimizer
from ..sampling import sample_robot_structures
from log_enabled_ppo import run_ppo_with_logging


def evaluate_robot_performance(
        structure: np.ndarray,
        env_name: str = "Walker-v0",
        model_save_dir: str = "models",
        model_save_name: str = "controller",
        ppo_args: Optional[argparse.Namespace] = None,
        seed: int = 42,
) -> float:
    os.makedirs(model_save_dir, exist_ok=True)

    args = ppo_args
    connections = get_full_connectivity(structure)

    score = run_ppo_with_logging(
        args=args,
        body=structure,
        connections=connections,
        env_name=env_name,
        model_save_dir=model_save_dir,
        model_save_name=model_save_name,
        seed=seed
    )
    return score


def _ppo_worker(
        idx: int,
        structure: np.ndarray,
        env_name: str,
        model_save_dir: str,
        ppo_args: argparse.Namespace,
        seed: int
) -> Tuple[int, float]:
    print(f"Evaluating robot {idx}...")
    score = evaluate_robot_performance(
        structure=structure,
        env_name=env_name,
        model_save_dir=model_save_dir,
        model_save_name=f"robot_{idx}",
        ppo_args=ppo_args,
        seed=seed + idx
    )
    return score


def batch_evaluate_robots(
        structures: np.ndarray,
        env_name: str = "Walker-v0",
        model_save_dir: str = "models",
        args: Optional[argparse.Namespace] = None,
        num_cores: int = 1,
        seed: int = 42,
) -> List[float]:
    import torch.multiprocessing as multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    import utils.mp_group as mp

    os.makedirs(model_save_dir, exist_ok=True)
    num_robots = len(structures)
    scores = [0.0] * num_robots

    print(f"Starting batch evaluation for {num_robots} robots on environment '{env_name}'.")
    start_time = time.time()

    if num_cores > 1:
        group = mp.Group()
        for i, structure in enumerate(structures):
            job_args = (i, structure, env_name, model_save_dir, args, seed)
            group.add_job(
                _ppo_worker,
                job_args,
                lambda score, idx=i: scores.__setitem__(idx, score)
            )
        group.run_jobs(num_cores)
    else:
        # Fallback to sequential evaluation
        for i, structure in enumerate(structures):
            job_args = (i, structure, env_name, model_save_dir, args, seed)
            score = _ppo_worker(*job_args)
            scores[i] = score

    duration = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
    print(f"Batch evaluation finished in {duration}.")
    print(f"Controllers saved in: {model_save_dir}")

    return scores


def generate_and_evaluate_robots(
        diffusion_model: SEDD,
        num_robots: int = 10,
        structure_shape: Tuple[int, int] = (5, 5),
        env_name: str = "Walker-v0",
        ppo_args: Optional[argparse.Namespace] = None,
        num_cores: int = 1,
        device: torch.device = torch.device("cpu"),
        output_dir: Optional[str] = None,
        task_optimizer: Optional[TaskOptimizer] = None,
        steps: int = 64,
) -> Tuple[np.ndarray, List[float]]:
    """Generates, evaluates, and saves a set of robots."""
    print("Starting Robot Generation and Evaluation")

    # Generate the initial robot structures
    print(f"Generating {num_robots} robot structures...")
    _, structures = sample_robot_structures(
        diffusion_model=diffusion_model,
        task_model=task_optimizer,
        batch_size=num_robots,
        structure_shape=structure_shape,
        steps=steps,
        device=device,
        enforce_validity=True
    )

    if structures.shape[0] == 0:
        print("Error: No valid structures were generated.")
        return np.array([]), []

    print(f"Successfully generated {structures.shape[0]} valid structures.")

    controller_dir = "models"
    if output_dir:
        struct_dir = os.path.join(output_dir, 'structure')
        controller_dir = os.path.join(output_dir, 'controller')
        os.makedirs(struct_dir, exist_ok=True)
        os.makedirs(controller_dir, exist_ok=True)

        # Save the generated structures
        for i, structure in enumerate(structures):
            path = os.path.join(struct_dir, f"robot_{i}.npz")
            np.savez(path, structure=structure)
        print(f"Saved {len(structures)} structures to '{struct_dir}'")

    # Evaluate the generated robots
    scores = batch_evaluate_robots(
        structures=structures,
        env_name=env_name,
        model_save_dir=controller_dir,
        args=ppo_args,
        num_cores=num_cores
    )

    # Save scores
    if output_dir:
        score_path = os.path.join(output_dir, 'scores.txt')
        with open(score_path, 'w') as f:
            for i, score in enumerate(scores):
                f.write(f"Robot {i}\tScore: {score}\n")
        print(f"Saved scores to '{score_path}'")

    print("Process Complete")
    return structures, scores
