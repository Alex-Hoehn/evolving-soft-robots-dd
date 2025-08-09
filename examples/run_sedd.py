import argparse
import datetime
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from evogym import is_connected

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJ_ROOT))

from sedd.models.task import TaskOptimizer
from sedd.run import train_diffusion, train_task
from sedd.sedd_logger import SEDDExperimentLogger
from sedd.utils.eval import batch_evaluate_robots, get_full_connectivity, sample_robot_structures
from ppo.args import add_ppo_args


def setup_generation_dirs(exp_dir, gen):
    gen_dir = exp_dir / f"generation_{gen}"
    struct_dir = gen_dir / "structure"
    ctrl_dir = gen_dir / "controller"
    struct_dir.mkdir(parents=True, exist_ok=True)
    ctrl_dir.mkdir(parents=True, exist_ok=True)
    return gen_dir, struct_dir, ctrl_dir


def save_structures(structures, struct_dir):
    for i, struct in enumerate(structures):
        connections = get_full_connectivity(struct)
        np.savez(struct_dir / f"{i}.npz", structure=struct, connections=connections)


def save_scores_for_visualizer(scores, gen_dir):
    sorted_scores = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    with open(gen_dir / "output.txt", "w") as f:
        for robot_id, score in sorted_scores:
            f.write(f"{robot_id}\t\t{score}\n")


def main():
    parser = argparse.ArgumentParser(description="2-stage SEDD Evogym implementation.")

    # Experiment and environment arguments
    parser.add_argument('--exp-name', type=str, default='test_sedd')
    parser.add_argument('--env-name', type=str, default='Walker-v0')
    parser.add_argument('--structure-shape', nargs=2, type=int, default=(5, 5))
    parser.add_argument('--cycles', type=int, default=10, help="Number of generations.")
    parser.add_argument('--num-robots', type=int, default=4, help="Number of robots per generation.")
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--num-cores', type=int, default=1)

    # Training arguments
    parser.add_argument('--num-epochs', type=int, default=5, help="Epochs for diffusion model training.")
    parser.add_argument('--fine-tune-epochs', type=int, default=5, help="Epochs for task model fine-tuning.")
    parser.add_argument('--max-samples', type=int, default=2000)
    parser.add_argument('--eval-episodes', type=int, default=5)

    # Sampling arguments
    parser.add_argument('--num-generated', type=int, default=500)
    parser.add_argument('--predictor', type=str, default='analytic', choices=['analytic', 'euler', 'none'])
    parser.add_argument('--sampling-steps', type=int, default=64)
    parser.add_argument('--sampling-eps', type=float, default=1e-5)
    parser.add_argument('--denoise', action='store_true', default=True)

    # Config
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--load-model', type=str, default=None)
    parser.add_argument('--load-task-model', type=str, default=None)
    parser.add_argument('--override', action='store_true', help="Delete and override existing experiment.")
    parser.add_argument('--quick-test', action='store_true', help="Run with small settings for a quick test.")

    add_ppo_args(parser)
    args = parser.parse_args()

    start_time = time.time()

    # Quick test run
    if args.quick_test:
        args.num_epochs = 5
        args.total_timesteps = 50_000
        args.eval_episodes = 5
        args.num_robots = 4
        args.cycles = 3
        args.num_cores = 2
        args.sampling_steps = 32

    # Set up experiment directory
    exp_dir = Path("saved_data") / args.exp_name
    if exp_dir.exists() and args.override:
        print(f"Deleting existing directory: {exp_dir}")
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger = SEDDExperimentLogger(args.exp_name, exp_dir)
    device = torch.device(args.device)

    # Train Base SEDD Model
    print("Training Base Model...")
    diffusion_model = train_diffusion(args)
    print("Base Model training complete")

    previous_scores = None
    task_model = None

    for gen in range(args.cycles):
        print(f"============== GENERATION {gen} ==============")
        logger.log_generation_start(gen)
        gen_dir, struct_dir, ctrl_dir = setup_generation_dirs(exp_dir, gen)

        # 1. Sample new robot structures
        # The first generation samples purely from the diffusion model. Following generations use the Task Model.
        sampling_kwargs = {
            'diffusion_model': diffusion_model,
            'task_model': task_model,
            'batch_size': args.num_robots,
            'structure_shape': tuple(args.structure_shape),
            'steps': args.sampling_steps,
            'device': device,
            'predictor': args.predictor,
            'enforce_validity': True,
            'max_attempts': 100,
        }

        if gen == 0:
            print("Sampling initial robot designs...")
        else:
            print("Improving robot designs...")
            task_ids = torch.zeros(args.num_robots, dtype=torch.long, device=device)
            score_tensor = torch.tensor(previous_scores[:args.num_robots], dtype=torch.float32, device=device).unsqueeze(1)
            sampling_kwargs.update({'task_id': task_ids, 'score': score_tensor})

        tokens, structures = sample_robot_structures(**sampling_kwargs)

        if structures.shape[0] == 0:
            print("Error: No valid structures generated.")
            break

        print(f"Generated {structures.shape[0]} new robot structures.")
        save_structures(structures, struct_dir)

        # 2. Evaluate the new robots
        print("Evaluating robot performance...")
        scores = batch_evaluate_robots(
            structures,
            env_name=args.env_name,
            model_save_dir=str(ctrl_dir),
            args=args,
            num_cores=args.num_cores,
            seed=gen * 123,
        )
        print(f"Evaluation complete. Best score: {max(scores):.4f}")
        save_scores_for_visualizer(scores, gen_dir)
        logger.log_generation_complete(gen, scores, task_model is not None)
        previous_scores = scores

        # 3. Train or fine-tune the task model
        if task_model is None:
            print("Training Task Model...")
        else:
            print("Fine-tuning Task Model...")
            args.load_task_model = str(exp_dir / "models/task/task_model_latest.pt")

        task_model = train_task(args, diffusion_model=diffusion_model)
        print("Task Model ready.")

    # Finalize Experiment
    logger.log_experiment_complete()
    end_time = time.time()
    duration = end_time - start_time
    h, rem = divmod(duration, 3600)
    m, s = divmod(rem, 60)
    formatted_time = f"{int(h)}h {int(m)}m {int(s)}s"

    print(f"Experiment complete. Total time: {formatted_time}")
    print(f"Results saved to: {exp_dir}")

    time_log_path = exp_dir / "training_time.txt"
    with open(time_log_path, "w") as f:
        f.write(f"Start Time: {datetime.datetime.fromtimestamp(start_time):%Y-%m-%d %H:%M:%S}\n")
        f.write(f"End Time:   {datetime.datetime.fromtimestamp(end_time):%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Duration:   {formatted_time} ({duration:.2f} seconds)\n")


if __name__ == "__main__":
    main()