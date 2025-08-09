import os
import torch
import yaml
import logging
from omegaconf import OmegaConf

import evogym.envs
from . import sedd_losses
from .ema import ExponentialMovingAverage
from .models.diffusion import DiscreteDiffusionModel
from .models.task import create_task_optimizer
from .utils.data import RobotStructureDataset, TaskRobotDataset, create_data_loaders
from .utils.train import train_diffusion_model, train_task_optimizer, load_model_checkpoint
from .utils.eval import generate_and_evaluate_robots


def setup_logging(log_dir) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "ddit.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger("ddit")

def load_config(path) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_config(cfg, path):
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

def create_vocab() -> dict:
    return {
        "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
        "<pad>": 5, "<mask>": 6, "<s>": 7, "</s>": 8, "<unk>": 9
    }

def create_default_config() -> dict:
    return {
        "model": {"hidden_size": 768, "cond_dim": 128, "length": 25, "n_blocks": 8,
                  "n_heads": 8, "scale_by_sigma": True, "dropout": 0.1},
        "training": {"batch_size": 16, "n_iters": 100_000, "accum": 1, "ema": 0.9999,
                     "snapshot_freq": 1000, "log_freq": 50, "eval_freq": 500},
        "noise": {"type": "loglinear", "sigma_min": 1e-4, "sigma_max": 20},
        "sampling": {"steps": 64, "enforce_connectivity": True},
        "optim": {"optimizer": "AdamW", "weight_decay": 0, "lr": 3e-4, "beta1": 0.9,
                  "beta2": 0.999, "eps": 1e-8, "warmup": 1000, "grad_clip": 1.0},
        "graph": {"type": "absorb"},
        "tokens": 10,
        "task_optim": {"total_timesteps": 100_000, "eval_episodes": 5, "num_cores": 4,
                       "visualize": True, "ema": 0.9999, "num_robots": 10, "model_save_dir": "models"}
    }

def train_diffusion(args):
    # Setup
    exp_dir = os.path.join('saved_data', args.exp_name)
    logger = setup_logging(exp_dir)
    logger.info(f"Starting diffusion training: {args.exp_name}")

    config = create_default_config()
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    config['training']['batch_size'] = args.batch_size
    config['task_optim'].update({
        'env_name': args.env_name,
        'structure_shape': list(args.structure_shape)
    })
    save_config(config, os.path.join(exp_dir, 'config.yaml'))
    device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    # Data
    vocab = create_vocab()
    data_dir = os.path.join(exp_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    dataset = RobotStructureDataset(
        data_dir, vocab, args.max_samples, args.structure_shape,
        generate_if_empty=True, num_generated=args.num_generated
    )
    train_loader, val_loader = create_data_loaders(dataset, config['training']['batch_size'], 0.9, 0)

    # Model
    model = DiscreteDiffusionModel(**config['model'], vocab_size=len(vocab)).to(device)
    if args.load_model:
        model, _, _ = load_model_checkpoint(model=model, checkpoint_path=args.load_model, device=device)

    ema = ExponentialMovingAverage(model.parameters(), decay=config['training']['ema'])
    optimizer = sedd_losses.get_optimizer(OmegaConf.create(config), model.parameters())
    diffusion_dir = os.path.join(exp_dir, 'models', 'diffusion')
    os.makedirs(diffusion_dir, exist_ok=True)

    # Train
    train_diffusion_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        num_epochs=args.num_epochs,
        device=device,
        save_dir=diffusion_dir,
        save_interval=config["training"]["snapshot_freq"],
        log_interval=config["training"]["log_freq"],
        ema=ema
    )
    logger.info("Diffusion training complete.")
    return model


def train_task(args, diffusion_model: DiscreteDiffusionModel = None):
    # Setup
    exp_dir = os.path.join("saved_data", args.exp_name)
    logger = setup_logging(exp_dir)
    logger.info(f"Starting task optimization: {args.exp_name}")

    config = load_config(os.path.join(exp_dir, "config.yaml"))
    config["training"]["batch_size"] = args.batch_size
    device = torch.device("cpu")
    vocab = create_vocab()

    gen_dir = os.path.join(exp_dir, "generation_0")
    ctrl_dir = os.path.join(gen_dir, "controller")
    task_dir = os.path.join(exp_dir, "models", "task")
    os.makedirs(ctrl_dir, exist_ok=True)
    os.makedirs(task_dir, exist_ok=True)

    if not os.path.exists(os.path.join(gen_dir, "output.txt")):
        logger.info("Generating new robot structures for task training...")
        if diffusion_model is None:
            logger.info("Loading pre-trained diffusion model...")
            diffusion_model = DiscreteDiffusionModel(**config["model"], vocab_size=len(vocab)).to(device)
            diffusion_ckpt = os.path.join(exp_dir, "models", "diffusion", "diffusion_model_latest.pt")
            diffusion_model, _, _ = load_model_checkpoint(model=diffusion_model, checkpoint_path=diffusion_ckpt, device=device)

        generate_and_evaluate_robots(
            diffusion_model=diffusion_model,
            num_robots=min(args.num_robots, args.num_generated),
            structure_shape=args.structure_shape,
            env_name=args.env_name,
            model_save_dir=ctrl_dir,
            args=args,
            num_cores=args.num_cores,
            device=device,
            save_dir=gen_dir,
            visualize=False,
        )

    # Data
    logger.info("Creating task dataset and data loaders...")
    task_dataset = TaskRobotDataset(exp_dir, vocab, args.max_samples, args.structure_shape)
    train_loader, val_loader = create_data_loaders(task_dataset, config["training"]["batch_size"], 0.9, 0)
    logger.info(f"Dataset ready: {len(task_dataset)} samples.")

    # Model
    logger.info("Creating task optimization model...")
    task_model = create_task_optimizer(**config["model"], vocab_size=len(vocab)).to(device)
    if args.load_task_model:
        logger.info(f"Loading pre-trained task model from {args.load_task_model}")
        task_model, _, _ = load_model_checkpoint(model=task_model, checkpoint_path=args.load_task_model, device=device)

    # Training setup
    ema = ExponentialMovingAverage(task_model.parameters(), decay=config['training']['ema'])
    optimizer = sedd_losses.get_optimizer(OmegaConf.create(config), task_model.parameters())
    normalize_score = lambda score: torch.clamp(score / 10.0, 0.0, 1.0)

    # Train
    logger.info("Starting task model training...")
    train_task_optimizer(
        model=task_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        num_epochs=args.num_epochs,
        device=device,
        save_dir=task_dir,
        save_interval=config["training"]["snapshot_freq"],
        log_interval=config["training"]["log_freq"],
        score_normalization=normalize_score,
        ema=ema
    )
    logger.info("Task optimization complete!")
    return task_model