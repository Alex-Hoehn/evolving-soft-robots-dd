import os
from typing import List, Optional
import numpy as np

import torch
import json

from ppo.eval import eval_policy
from stable_baselines3.common.callbacks import BaseCallback

class EvalCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(
            self,
            body: np.ndarray,
            env_name: str,
            eval_every: int,
            n_evals: int,
            n_envs: int,
            model_save_dir: str,
            model_save_name: str,
            exp_name,
            generation,
            connections: Optional[np.ndarray] = None,
            verbose: int = 0
    ):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

        self.body = body
        self.connections = connections
        self.env_name = env_name
        self.eval_every = eval_every
        self.n_evals = n_evals
        self.n_envs = n_envs
        self.model_save_dir = model_save_dir
        self.model_save_name = model_save_name
        self.exp_name = exp_name
        self.generation = generation

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        # Path for the eval log file
        self.eval_log_path = os.path.join(model_save_dir, f'{model_save_name}.log')

        # Path for the overall evals file
        self.exp_root_dir = os.path.join("saved_data", exp_name)
        self.evals_json_path = os.path.join(self.exp_root_dir, "extracted_evaluations.json")

        self.timesteps = []
        self.eval_results = []
        self.best_reward = -float('inf')

        if os.path.exists(self.eval_log_path):
            os.remove(self.eval_log_path)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """

        if self.num_timesteps % self.eval_every == 0:
            self._validate_and_save()
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self._validate_and_save()

        model_path = os.path.join(self.model_save_dir, f"{self.model_save_name}.zip")
        self.model.save(model_path)

    def _validate_and_save(self) -> None:
        rewards = eval_policy(
            model=self.model,
            body=self.body,
            connections=self.connections,
            env_name=self.env_name,
            n_evals=self.n_evals,
            n_envs=self.n_envs,
        )

        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        min_reward = float(np.min(rewards))
        max_reward = float(np.max(rewards))

        self.timesteps.append(int(self.num_timesteps))
        self.eval_results.append(mean_reward)

        out = f"[{self.model_save_name}] Time: {self.num_timesteps}, Mean: {mean_reward:.3f}, Std: {std_reward:.3f}, Min: {min_reward:.3f}, Max: {max_reward:.3f}"

        if mean_reward > self.best_reward:
            out += f" NEW BEST ({mean_reward:.3} > {self.best_reward:.3})"
            self.best_reward = mean_reward
            model_path = os.path.join(self.model_save_dir, f"{self.model_save_name}.zip")
            self.model.save(model_path)

        if self.verbose > 0:
            print(out)

        with open(self.eval_log_path, 'a') as f:
            f.write(f"timestep: {self.num_timesteps}, reward: {mean_reward}\n")

        self._update_evals_json()

    def _update_evals_json(self) -> None:
        robot_id = f"{self.generation}_{self.model_save_name}"

        if os.path.exists(self.evals_json_path):
            with open(self.evals_json_path, 'r') as f:
                data = json.load(f)
        else:
            data = {
                "robot_evaluations": {},
                "generation_data": {
                    "generation_nums": [],
                    "best_rewards": []
                },
                "best_over_time": {
                    "timesteps": [],
                    "best_rewards": []
                }
            }

        data["robot_evaluations"][robot_id] = {
            "timesteps": self.timesteps.copy(),
            "rewards": self.eval_results.copy()
        }

        self._update_generation_data(data)

        self._update_best_over_time(data)

        with open(self.evals_json_path, 'w') as f:
            json.dump(data, f, indent=4)

    def _update_generation_data(self, data: dict) -> None:
        if self.generation not in data["generation_data"]["generation_nums"]:
            data["generation_data"]["generation_nums"].append(self.generation)
            data["generation_data"]["best_rewards"].append(-float('inf'))

        gen_index = data["generation_data"]["generation_nums"].index(self.generation)
        current_gen_best = data["generation_data"]["best_rewards"][gen_index]

        if self.best_reward > current_gen_best:
            data["generation_data"]["best_rewards"][gen_index] = self.best_reward

    def _update_best_over_time(self, data: dict) -> None:
        all_timestep_reward_pairs = []

        for robot_id, robot_data in data["robot_evaluations"].items():
            for timestep, reward in zip(robot_data["timesteps"], robot_data["rewards"]):
                all_timestep_reward_pairs.append((timestep, reward))

        all_timestep_reward_pairs.sort(key=lambda x: x[0])

        best_over_time_timesteps = []
        best_over_time_rewards = []
        current_best = -float('inf')

        for timestep, reward in all_timestep_reward_pairs:
            if reward > current_best:
                current_best = reward

            if (not best_over_time_timesteps or
                    timestep != best_over_time_timesteps[-1] or
                    current_best != best_over_time_rewards[-1]):
                best_over_time_timesteps.append(timestep)
                best_over_time_rewards.append(current_best)

        data["best_over_time"]["timesteps"] = best_over_time_timesteps
        data["best_over_time"]["best_rewards"] = best_over_time_rewards
