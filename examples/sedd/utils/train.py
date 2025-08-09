import torch
import numpy as np
import os
import logging
from tqdm import tqdm
from typing import Dict, List, Optional, Callable

from ..ema import ExponentialMovingAverage
from ..models.diffusion import SEDD
from ..models.task import TaskOptimizer
from .. import graph_lib, noise_lib, sedd_losses


def prepare_batch(batch, device):
    if isinstance(batch, dict):
        tokens = batch["tokens"].to(device)
        score = batch.get("score", None)
        if score is not None:
            return tokens.view(tokens.size(0), -1), score.to(device)
        return tokens.view(tokens.size(0), -1), None
    else:
        tokens = batch.to(device)
        return tokens.view(tokens.size(0), -1), None


def train_diffusion_model(
        model: SEDD,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_epochs: int = 100,
        device: torch.device = torch.device("cpu"),
        save_dir: str = "checkpoints",
        save_interval: int = 10,
        log_interval: int = 10,
        ema: Optional[ExponentialMovingAverage] = None,
) -> Dict[str, List[float]]:
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("diffusion_training")
    model.to(device)

    # Setup SEDD components
    graph = graph_lib.get_graph(model.config, device)
    noise = noise_lib.get_noise(model.config).to(device)
    optimize_fn = sedd_losses.optimization_manager(model.config)
    train_step_fn = sedd_losses.get_step_fn(noise, graph, True, optimize_fn, 1)
    eval_step_fn = sedd_losses.get_step_fn(noise, graph, False, optimize_fn, 1)

    state = {
        'model': model,
        'optimizer': optimizer,
        'scaler': torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda')),
        'step': 0,
        'ema': ema,
    }
    history = {"train_loss": [], "val_loss": [], "epochs": []}

    logger.info(f"Starting diffusion model training for {num_epochs} epochs.")

    for epoch in range(1, num_epochs + 1):
        # Training Phase
        model.train()
        train_losses = []
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training")):
            tokens, _ = prepare_batch(batch, device)
            loss = train_step_fn(state, tokens)
            if loss is not None:
                train_losses.append(loss.item())
                if batch_idx % log_interval == 0:
                    logger.info(f"Batch {batch_idx}, Loss: {loss.item():.6f}")

        if scheduler:
            scheduler.step()

        # Validation Phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                tokens, _ = prepare_batch(batch, device)
                loss = eval_step_fn(state, tokens)
                if loss is not None:
                    val_losses.append(loss.item())

        # Logging and Saving
        avg_train_loss = np.mean(train_losses) if train_losses else 0
        avg_val_loss = np.mean(val_losses) if val_losses else 0
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["epochs"].append(epoch)

        logger.info(f"Epoch {epoch} Summary | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if epoch % save_interval == 0 or epoch == num_epochs:
            checkpoint_path = os.path.join(save_dir, "diffusion_model_latest.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    return history


def train_task_optimizer(
        model: TaskOptimizer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_epochs: int = 100,
        device: torch.device = torch.device("cpu"),
        save_dir: str = "checkpoints",
        save_interval: int = 10,
        log_interval: int = 10,
        score_normalization: Optional[Callable] = None,
        ema: Optional[ExponentialMovingAverage] = None,
) -> Dict[str, List[float]]:
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("task_optimizer_training")
    model.to(device)

    # Setup SEDD components
    graph = graph_lib.get_graph(model.config, device)
    noise = noise_lib.get_noise(model.config).to(device)
    normalize_score = score_normalization or (lambda x: x)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    history = {"train_loss": [], "val_loss": [], "epochs": []}

    logger.info(f"Starting task optimizer training for {num_epochs} epochs.")

    for epoch in range(1, num_epochs + 1):
        # Training Phase
        model.train()
        train_losses = []
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training")):
            tokens, scores = prepare_batch(batch, device)
            scores = normalize_score(scores)

            # Sample noise level and perturb the data
            t = torch.rand(tokens.size(0), device=device) * (1 - 1e-3) + 1e-3
            sigma, dsigma = noise(t)
            perturbed = graph.sample_transition(tokens, sigma[:, None])

            # Calculate loss
            log_score = model(perturbed, scores, sigma)
            loss = graph.score_entropy(log_score, sigma[:, None], perturbed, tokens)
            loss = ((dsigma[:, None] * loss).sum(-1)).mean()

            # Backpropagation
            if torch.isfinite(loss):
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if ema:
                    ema.update(model.parameters())
                train_losses.append(loss.item())

            if (batch_idx + 1) % log_interval == 0 and train_losses:
                logger.info(f"Batch {batch_idx+1}, Avg Loss: {np.mean(train_losses):.6f}")


        if scheduler:
            scheduler.step()

        # Validation Phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            if ema:
                ema.store(model.parameters())
                ema.copy_to(model.parameters())

            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                tokens, scores = prepare_batch(batch, device)
                scores = normalize_score(scores)

                t = torch.rand(tokens.size(0), device=device) * (1 - 1e-3) + 1e-3
                sigma, dsigma = noise(t)
                perturbed = graph.sample_transition(tokens, sigma[:, None])

                log_score = model(perturbed, scores, sigma)
                loss = graph.score_entropy(log_score, sigma[:, None], perturbed, tokens)
                loss = ((dsigma[:, None] * loss).sum(-1)).mean()

                if torch.isfinite(loss):
                    val_losses.append(loss.item())

            if ema:
                ema.restore(model.parameters())

        # Logging and Saving
        avg_train_loss = np.mean(train_losses) if train_losses else 0
        avg_val_loss = np.mean(val_losses) if val_losses else 0
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["epochs"].append(epoch)

        logger.info(f"Epoch {epoch} Summary | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if epoch % save_interval == 0 or epoch == num_epochs:
            checkpoint_path = os.path.join(save_dir, "task_model_latest.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    return history


def load_model_checkpoint(
        model: torch.nn.Module,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device("cpu"),
        ema: Optional[ExponentialMovingAverage] = None,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if ema and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])

    print(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint.get('epoch', 'N/A')})")
    return model, optimizer, checkpoint
