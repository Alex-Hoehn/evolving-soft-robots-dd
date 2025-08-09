import argparse
from typing import Optional
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from ppo.eval import eval_policy
from ppo.custom_callback import EvalCallback

import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"]=""

def run_ppo_with_logging(
        args: argparse.Namespace,
        body: np.ndarray,
        env_name: str,
        model_save_dir: str,
        model_save_name: str,
        connections: Optional[np.ndarray] = None,
        seed: int = 42,
) -> float:
    """
    Run ppo, log the results and return the best reward achieved during evaluation.
    """

    model_save_dir = str(model_save_dir)
    model_save_name = str(model_save_name)

    path_parts = model_save_dir.split(os.sep)
    generation = None
    for part in path_parts:
        if part.startswith("generation_"):
            generation = int(part.split("_")[1])
            break

    if generation is None:
        raise ValueError(f"Could not extract generation number from path: {model_save_dir}")

    # Parallel environments
    vec_env = make_vec_env(env_name, n_envs=args.n_envs, seed=seed, env_kwargs={
        'body': body,
        'connections': connections,
    })

    os.makedirs(model_save_dir, exist_ok=True)

    # Custom EvalCallback
    callback = EvalCallback(
        body=body,
        connections=connections,
        env_name=env_name,
        eval_every=args.eval_interval,
        n_evals=args.n_evals,
        n_envs=args.n_eval_envs,
        model_save_dir=model_save_dir,
        model_save_name=model_save_name,
        exp_name=args.exp_name,
        generation=generation,
        verbose=args.verbose_ppo,
    )

    # Train
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=args.verbose_ppo,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range
    )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        log_interval=args.log_interval
    )

    controller_path = os.path.join(model_save_dir, f"{model_save_name}.zip")
    if not os.path.exists(controller_path):
        model.save(controller_path)

    return callback.best_reward