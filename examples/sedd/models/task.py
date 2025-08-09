# adapted from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, Optional, List, Union, Any
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf
from torch import Tensor


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
            # This makes the transformation on v an identity.
            self.cos_cached[:,:,2,:,:].fill_(1.)
            self.sin_cached[:,:,2,:,:].fill_(0.)

        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=-1
    )


@torch.jit.script
def _apply_rotary_pos_emb_torchscript(qkv, cos, sin):
    return (qkv * cos) + (rotate_half(qkv) * sin)


def apply_rotary_pos_emb(qkv, cos, sin):
    try:
        import flash_attn.layers.rotary
        cos = cos[0,:,0,0,:cos.shape[-1]//2]
        sin = sin[0,:,0,0,:sin.shape[-1]//2]
        return flash_attn.layers.rotary.apply_rotary_emb_qkv_(
            qkv, cos, sin
        )
    except:
        return _apply_rotary_pos_emb_torchscript(qkv, cos, sin)


def modulate_fused(x, shift, scale):
    return x * (1 + scale) + shift


def bias_dropout_add_scale(
        x: Tensor, bias: Optional[Tensor], scale: Tensor, residual: Optional[Tensor], prob: float, training: bool
) -> Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)

    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(x, bias, scale, residual, prob, training)

    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_scale_fused_train(
        x: Tensor, bias: Optional[Tensor], scale: Tensor, residual: Optional[Tensor], prob: float
) -> Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
        x: Tensor, bias: Optional[Tensor], scale: Tensor, residual: Optional[Tensor], prob: float
) -> Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)

        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )

        qkv = rearrange(qkv, 'b s ... -> (b s) ...')

        # Flash attention with fallback
        if seqlens is None:
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=qkv.device
            )
        else:
            cu_seqlens = seqlens.cumsum(-1)

        try:
            from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
            x = flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, seq_len, 0., causal=False)
            x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
        except:
            # Fallback to standard attention
            qkv = rearrange(qkv, '(b s) three h d -> b s three h d', b=batch_size)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)  # (b, h, s, d)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1)), dim=-1)
            x = torch.matmul(attn, v).transpose(1, 2)  # (b, s, h, d)
            x = rearrange(x, 'b s h d -> b s (h d)')

        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None, gate_mlp, x, self.dropout
        )

        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors,
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TaskOptimizer(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()

        # hack to make loading in configs easier
        if type(config) == dict:
            config = OmegaConf.create(config)

        self.config = config

        self.absorb = config.graph.type == "absorb"
        vocab_size = config.tokens + (1 if self.absorb else 0)

        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        self.rotary_emb = Rotary(config.model.hidden_size // config.model.n_heads)

        # Score embedding for conditioning (change from the original codebase)
        self.score_embed = nn.Sequential(
            nn.Linear(1, config.model.cond_dim),
            nn.SiLU(),
            nn.Linear(config.model.cond_dim, config.model.cond_dim),
        )

        self.blocks = nn.ModuleList([
            DDiTBlock(config.model.hidden_size, config.model.n_heads, config.model.cond_dim, dropout=config.model.dropout)
            for _ in range(config.model.n_blocks)
        ])

        self.output_layer = DDitFinalLayer(config.model.hidden_size, vocab_size, config.model.cond_dim)
        self.scale_by_sigma = config.model.scale_by_sigma

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def forward(self, indices, score, sigma):
        x = self.vocab_embed(indices)

        # Basic sigma conditioning (matching original)
        sigma_c = F.silu(self.sigma_map(sigma))

        # Score conditioning (only addition)
        if score is None:
            # Unconditional (classifier-free guidance)
            score_c = torch.zeros_like(sigma_c)
        else:
            score_c = self.score_embed(score)

        # Simple addition (no complex integration)
        c = sigma_c + score_c

        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)

            x = self.output_layer(x, c)

        if self.scale_by_sigma:
            assert self.absorb, "Haven't configured this to work."
            esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(x.dtype)[:, None, None]
            x = x - esigm1_log - np.log(x.shape[-1] - 1)# this will be approximately averaged at 0

        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))

        return x

    def get_loss(self, x, y, score, sigma=None):
        """
        Calculate training loss for the task optimizer.
        """
        # Sample sigma if not provided (needed for SEDD)
        if sigma is None:
            sigma = torch.rand(x.size(0), device=x.device) * 19.0 + 1e-4

        # Detach inputs to prevent gradient accumulation
        x = x.detach() if x.requires_grad else x
        y = y.detach() if y.requires_grad else y

        try:
            # Forward pass with sigma
            logits = self(x, score, sigma)

            # Check for invalid logits before loss computation
            if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
                print("⚠️  Invalid logits detected, returning small loss")
                return torch.tensor(0.001, device=x.device, requires_grad=True)

            # Clamp logits to avoid overflow in softmax
            logits = torch.clamp(logits, min=-50, max=50)

            # Ensure y is within valid range
            if (y < 0).any() or (y >= logits.shape[-1]).any():
                print("❌ Invalid target token(s) in y:", y.unique())
                return torch.tensor(0.001, device=x.device, requires_grad=True)

            # Calculate cross entropy loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1),
                reduction='mean'
            )

            # Check for non-finite loss
            if not torch.isfinite(loss):
                print("💥 Loss became non-finite (inf or NaN)!")
                print("  → score:", score.flatten().tolist() if hasattr(score, 'flatten') else score)
                print("  → y min/max:", y.min().item(), "/", y.max().item())
                print("  → logits stats: min {:.2f}, max {:.2f}".format(
                    logits.min().item(), logits.max().item()
                ))
                return torch.tensor(0.001, device=x.device, requires_grad=True)

            return loss

        except Exception as e:
            print(f"❌ Error in get_loss: {e}")
            return torch.tensor(0.001, device=x.device, requires_grad=True)

    @torch.no_grad()
    def improve(
            self,
            x: torch.Tensor,      # [B, L]   tokens or MASK grid
            score: torch.Tensor,  # [B, 1]
            *,
            steps: int = 64,
            sigma_min: float = 1e-4,
            sigma_max: float = 20.0,
            temperature: float = 1.3,
            top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Refine a batch of robots by running the same σ-schedule denoising loop
        the diffusion generator uses, but conditioned on score.

        * Handles both `Tensor` **and** `(logits, …)` tuples returned from
          `forward`, so it can work with or without auxiliary outputs.
        """

        # Simple temperature sampling (matching original pattern)
        def _sample_with_temperature(logits_: torch.Tensor,
                                     temperature_: float = 1.3) -> torch.Tensor:

            if isinstance(logits_, (tuple, list)):
                logits_ = logits_[0]  # unwrap

            if temperature_ != 1.0:
                logits_ = logits_ / temperature_

            probs = torch.softmax(logits_, -1)
            idx = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
            return idx.view(probs.shape[:-1])

        # denoising loop (matching original structure)
        device = x.device
        batch_size = x.size(0)
        sigmas = torch.linspace(sigma_max, sigma_min, steps, device=device)

        tokens = torch.full_like(x, fill_value=self.vocab_size) if self.absorb else x.clone()

        for sigma in sigmas:
            logits = self(tokens, score, sigma.expand(batch_size))

            # unwrap if model returned (logits, aux)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            tokens = _sample_with_temperature(logits, temperature)

        tokens = tokens.clone()
        tokens[tokens > 4] = 0

        return tokens


def create_task_optimizer(
        hidden_size: int = 768,
        n_blocks: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        length: int = 25,
        vocab_size: int = 10,
        cond_dim: int = 128,
        scale_by_sigma: bool = True,
):
    config_dict = {
        'model': {
            'hidden_size': hidden_size,
            'n_blocks': n_blocks,
            'n_heads': n_heads,
            'dropout': dropout,
            'length': length,
            'scale_by_sigma': scale_by_sigma,
            'cond_dim': cond_dim,
        },
        'tokens': vocab_size,
        'graph': {'type': 'absorb'}
    }
    config = OmegaConf.create(config_dict)
    return TaskOptimizer(config)