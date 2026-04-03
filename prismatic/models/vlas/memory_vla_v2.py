"""
memory_vla_v2.py

MemoryVLA_V2: Replaces the retrieval-based memory approach with an explicit causal context transformer
over a fixed-length window of past cognitive tokens. This version generates action tokens directly from
the CausalContextTransformer output (similar to OpenVLA), rather than using a separate action head.

Key features:
- forward() accepts input_ids and pixel_values as *lists* of length T (one per timestep).
- All hidden states at timesteps 0..T-2 are detached; only the last (current) timestep retains gradients.
- CausalContextTransformer processes the sequence of T cognitive tokens and produces an enriched token.
- The enriched token is used to generate action token logits directly via a prediction head.
- predict_action() maintains a buffer of past cog_tokens across calls and resets on episode boundaries.
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers import LlamaTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.vlms.prismatic import PrismaticVLM, IGNORE_INDEX
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.action_tokenizer import ActionTokenizer

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


overwatch = initialize_overwatch(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Output dataclass for MemoryVLA_V2
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryVLAOutput:
    """Output class for MemoryVLA_V2 forward pass."""
    loss: torch.Tensor
    context_token: torch.Tensor
    last_vlm_output: CausalLMOutputWithPast
    logits: Optional[torch.Tensor] = None  # [B, num_action_tokens, vocab_size] from CausalContextTransformer

    def __getitem__(self, key):
        """Allow dictionary-style access for backward compatibility."""
        return getattr(self, key)


# ──────────────────────────────────────────────────────────────────────────────
# Utility: Index at timestep
# ──────────────────────────────────────────────────────────────────────────────

def index_at_timestep(data, t: int):
    """
    Index input at timestep t.
    
    Handles both:
    - Single tensors:     tensor[:, t, ...]
    - Dict of tensors:    {k: v[:, t, ...] for k, v in data.items()}
    
    Args:
        data: torch.Tensor or Dict[str, torch.Tensor] with time dimension at position 1
        t:    Timestep index
    
    Returns:
        Indexed tensor or dict of tensors (time dimension removed)
    """
    if isinstance(data, dict):
        return {k: v[:, t, ...] for k, v in data.items()}
    else:
        return data[:, t, ...]


def get_time_dim(data):
    """
    Get the time dimension from input data.
    
    Handles both:
    - Single tensors:     returns tensor.shape[1]
    - Dict of tensors:    returns shape[1] and asserts all tensors have same time dim
    
    Args:
        data: torch.Tensor or Dict[str, torch.Tensor] with time dimension at position 1
    
    Returns:
        Time dimension (int) at position 1
    """
    if isinstance(data, dict):
        time_dims = {}
        for key, val in data.items():
            if len(val.shape) >= 2:
                time_dims[key] = val.shape[1]
        
        # Assert all time dimensions are the same
        if len(time_dims) > 0:
            first_key = next(iter(time_dims.keys()))
            first_time_dim = time_dims[first_key]
            for key, t_dim in time_dims.items():
                assert t_dim == first_time_dim, (
                    f"Time dimension mismatch: {key} has {t_dim}, expected {first_time_dim}"
                )
            return first_time_dim
        return 1
    else:
        assert len(data.shape) >= 2, (
            f"Tensor must have at least 2 dimensions, got {data.shape}"
        )
        return data.shape[1]


# ──────────────────────────────────────────────────────────────────────────────
# CausalContextTransformer: Processes cognitive tokens causally
# ──────────────────────────────────────────────────────────────────────────────

class CausalContextTransformer(nn.Module):
    """
    A causal transformer that processes a window of T cognitive tokens and
    outputs a single context-enriched token corresponding to the last (current)
    position.

    Padding:
        When fewer than `context_length` tokens are available (e.g., at the start
        of an episode), the sequence is *left-padded* with a fixed learnable
        padding token so that the causal attention always sees a full-length
        sequence; the current token still attends only to tokens that came before
        or at the same position (i.e., it cannot see "future" pads since there are
        none -- all pads are in the past positions).

    Args:
        token_dim:        Dimensionality of each cognitive token.
        context_length:   Maximum number of timesteps in the context window.
        num_heads:        Number of attention heads.
        num_layers:       Number of transformer layers.
        dropout:          Dropout probability.
        embedding_layer:  LLM embedding layer for converting token IDs to embeddings.
        vocab_size:       Vocabulary size for output projection.
    """

    def __init__(
        self,
        token_dim: int,
        context_length: int,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.0,
        vocab_size: Optional[int] = None,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.context_length = context_length
        self.vocab_size = vocab_size
        self._embedding_layer = None  # Will be set by parent model

        # Learnable padding token
        self.pad_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        nn.init.normal_(self.pad_token, std=0.02)

        # Learnable positional embedding for the context window + label tokens
        # We need extra positions for label tokens that will be appended
        max_seq_len = context_length + 100  # Reserve space for label tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, token_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Causal transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(token_dim)
        
        # Output projection to vocabulary
        if vocab_size is not None:
            self.output_proj = nn.Linear(token_dim, vocab_size)
        else:
            self.output_proj = None

    def forward(
        self, 
        tokens: torch.Tensor, 
        labels: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            tokens: [B, T, D] where T <= context_length
                    The last position in T is the *current* timestep.
                    All earlier positions are historical (detached outside this fn).
            labels: [B, num_label_tokens] - action token IDs for current timestep
        
        Returns:
            out: [B, 1, D] – the output at the last (current) position.
            loss: Optional[torch.Tensor] - cross-entropy loss if labels provided
        """
        B, T, D = tokens.shape
        assert D == self.token_dim, f"Expected token_dim={self.token_dim}, got {D}"
        assert T <= self.context_length, (
            f"Sequence length {T} exceeds context_length {self.context_length}"
        )

        # Left-pad with learnable pad token so total length == context_length
        pad_len = self.context_length - T
        if pad_len > 0:
            pad_tokens = self.pad_token.expand(B, pad_len, D)
            tokens = torch.cat([pad_tokens, tokens], dim=1)  # [B, context_length, D]
        
        loss = None
        label_embeddings = None
        num_label_tokens = 0
        
        # If labels are provided, convert them to embeddings and append
        if labels is not None and self._embedding_layer is not None:
            # Flatten labels if multi-dimensional: [B, ...] -> [B, num_label_tokens]
            if labels.dim() > 2:
                labels_flat = labels.view(B, -1)
            else:
                labels_flat = labels
            
            num_label_tokens = labels_flat.shape[1]
            
            # Convert token IDs to embeddings - access dynamically
            label_embeddings = self._embedding_layer()(labels_flat)  # [B, num_label_tokens, D]
            
            # Append label embeddings to tokens
            tokens = torch.cat([tokens, label_embeddings], dim=1)  # [B, context_length + num_label_tokens, D]
        
        seq_len = tokens.shape[1]

        # Add positional embedding
        tokens = tokens + self.pos_embed[:, :seq_len, :]  # [B, seq_len, D]

        # Generate explicit causal mask and pass is_causal=False (mask encodes causality)
        # Passing is_causal=True without a mask raises RuntimeError in PyTorch's MHA
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=tokens.device, dtype=tokens.dtype
        )
        # causal_mask = torch.full((seq_len, seq_len), float("-inf"), dtype=tokens.dtype, device=tokens.device)
        # causal_mask = torch.triu(
        #     causal_mask,
        #     diagonal=1,
        # )
        x = self.transformer(tokens, mask=causal_mask, is_causal=False)
        out = self.out_norm(x)  # [B, seq_len, D]
        
        # Compute loss if labels were provided
        if labels is not None and self.output_proj is not None:
            # Get hidden states for label positions
            label_hidden = out[:, -num_label_tokens-1:-1, :]  # [B, num_label_tokens, D] # TODO check if we are getting the correct positions of tokens (actions only)
            
            # Project to vocabulary
            logits = self.output_proj(label_hidden)  # [B, num_label_tokens, vocab_size]
            
            # Compute cross-entropy loss
            # Predict next token, so labels[1:] are targets for logits[:-1]
            if num_label_tokens > 1:
                logits_for_loss = logits[:, :, :].contiguous()  # [B, num_label_tokens, vocab_size]
                labels_for_loss = labels_flat[:, :].contiguous()  # [B, num_label_tokens]
            else:
                # Single label token - no shifting possible, compute loss directly
                logits_for_loss = logits  # [B, 1, vocab_size]
                labels_for_loss = labels_flat  # [B, 1]
            
            loss = nn.functional.cross_entropy(
                logits_for_loss.view(-1, logits_for_loss.shape[-1]),
                labels_for_loss.view(-1),
                ignore_index=IGNORE_INDEX,
            )

        # Return only the last position of context (before label positions)
        context_out = out[:, self.context_length - 1:self.context_length, :]  # [B, 1, D]
        return context_out, loss, logits if labels is not None else None



# ──────────────────────────────────────────────────────────────────────────────
# MemoryVLA_V2
# ──────────────────────────────────────────────────────────────────────────────

class MemoryVLA_V2(PrismaticVLM):
    """
    MemoryVLA_V2 uses a CausalContextTransformer to consolidate cognitive tokens
    from multiple timesteps and generates action tokens directly (without a separate
    diffusion-based action head).

    Training forward() contract:
        input_ids:    Tensor[B, T, seq_len]  – T timesteps
        pixel_values: Tensor[B, T, C, H, W] or Dict[str, Tensor[B, T, ...]]
        attention_mask: Optional[Tensor[B, T, seq_len]]
        labels:       Tensor[B, action_dim] – action token IDs for current timestep only
        
    Inference predict_action() contract:
        - Accepts a single image and instruction per call
        - Maintains self._cog_token_buffer of past cognitive tokens
        - Resets buffer on episode boundaries (via unnorm_key parameter)
    """

    def __init__(
        self,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        norm_stats: Optional[Dict] = None,
        action_tokenizer: Optional[ActionTokenizer] = None,
        # CausalContextTransformer config
        context_length: int = 16,
        input_seq_len: int = 256,
        causal_num_heads: int = 8,
        causal_num_layers: int = 4,
        causal_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            model_id=model_id,
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            **kwargs,
        )
        
        self.norm_stats = norm_stats or {}
        self.action_tokenizer = action_tokenizer
        self.context_length = context_length
        self.input_seq_len = input_seq_len
        
        # Get cognitive token dimension from LLM backbone
        self.cog_token_size = llm_backbone.embed_dim
        
        # Get vocab size from LLM
        vocab_size = llm_backbone.llm.config.vocab_size
        
        # Compute the effective context length for CausalContextTransformer:
        # (context_length - 1) slots for past cognitive tokens (one per historical timestep)
        # + (num_patches + input_seq_len) slots for all tokens of the current timestep
        num_patches = self._get_num_patches()
        causal_context_length = (context_length - 1) + num_patches + input_seq_len
        
        # CausalContextTransformer for processing cognitive tokens
        self.causal_ctx_transformer = CausalContextTransformer(
            token_dim=self.cog_token_size,
            context_length=causal_context_length,
            num_heads=causal_num_heads,
            num_layers=causal_num_layers,
            dropout=causal_dropout,
            vocab_size=vocab_size,
        )
        
        # Set embedding layer accessor as a callable that accesses it dynamically
        # This ensures it works correctly after FSDP wrapping
        self.causal_ctx_transformer._embedding_layer = lambda: self.llm_backbone.llm.get_input_embeddings()
        
        # Inference-time buffer: list of detached cog tokens [1, 1, D]
        self._infer_cog_buffer: List[torch.Tensor] = []
        
        # Update module keys for checkpointing
        self.all_module_keys.extend(["causal_ctx_transformer"])
        self.trainable_module_keys.extend(["causal_ctx_transformer"])

    def _get_num_patches(self) -> int:
        """Return the number of visual patch tokens for the active vision backbone."""
        vb = self.vision_backbone
        if hasattr(vb, "featurizer") and vb.featurizer is not None:
            return vb.featurizer.patch_embed.num_patches
        if hasattr(vb, "siglip_featurizer") and vb.siglip_featurizer is not None:
            return vb.siglip_featurizer.patch_embed.num_patches
        raise ValueError("No featurizer found on vision backbone")

    def _extract_cog_token(
        self,
        hidden_state: torch.Tensor,   # [B, num_tokens, token_size]
        attention_mask: torch.Tensor,  # [B, num_text_tokens]
        num_patch: int,
    ) -> torch.Tensor:
        """Extract the single EOS / last-attention cognitive token from hidden_state."""
        # Drop visual tokens
        hs = hidden_state[:, num_patch:]  # [B, num_text_tokens, token_size]
        cumsum = attention_mask.cumsum(dim=1)  # [B, num_text_tokens]
        last_idx = (cumsum == cumsum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)  # [B,]
        expanded = last_idx.unsqueeze(-1).expand(-1, hs.size(-1))  # [B, token_size]
        cog = hs.gather(1, expanded.unsqueeze(1))  # [B, 1, token_size]
        return cog

    def forward(
        self,
        # Temporal inputs: [B, T, ...] where T is the number of timesteps
        input_ids: torch.LongTensor,                              # [B, T, seq_len]
        pixel_values: Union[torch.FloatTensor, Dict],             # [B, T, C, H, W] or Dict
        attention_mask: Optional[torch.Tensor] = None,            # [B, T, seq_len]
        labels: Optional[torch.LongTensor] = None,                # [B, chunk_size,action_dim] - action tokens for current timestep
        # Standard VLM kwargs
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> MemoryVLAOutput:
        """
        Process T timesteps through the VLM, extract cognitive tokens, consolidate
        via CausalContextTransformer, and generate action token logits.

        Returns:
            MemoryVLAOutput with:
                loss            - cross-entropy loss for action token prediction
                context_token   - enriched cognitive token from CausalContextTransformer
                last_vlm_output - VLM CausalLMOutput for the current (last) timestep
        """
        assert len(input_ids.shape) == 3, (
            "input_ids must be a 3D tensor of shape [B, T, seq_len]"
        )
        
        T = get_time_dim(input_ids)  # number of context timesteps
        B = input_ids.shape[0]
        
        # If attention_mask not provided, create it
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Collect cognitive tokens for each timestep
        cog_tokens = []
        num_patch = self._get_num_patches()
        
        for t in range(T):
            # Extract data for timestep t
            input_ids_t = index_at_timestep(input_ids, t)          # [B, seq_len]
            pixel_values_t = index_at_timestep(pixel_values, t)    # [B, C, H, W] or Dict
            attention_mask_t = index_at_timestep(attention_mask, t)  # [B, seq_len]
            
            # Run VLM forward for this timestep (with output_hidden_states=True)
            with torch.set_grad_enabled(t == T - 1):  # Only compute gradients for last timestep
                output_t = super().forward(
                    input_ids=input_ids_t,
                    pixel_values=pixel_values_t,
                    attention_mask=attention_mask_t,
                    labels=None,  # Don't compute VLM loss here
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            # Extract cognitive token from last hidden state
            # hidden_states is a tuple of (num_layers + 1) tensors, each [B, seq_len, D]
            last_hidden = output_t.hidden_states[-1]  # [B, seq_len, D]
            
            if t == T - 1:
                # For the current (last) timestep, use all vision and language tokens
                # so the CausalContextTransformer receives full context: [B, num_patches + seq_len, D]
                cog_token_t = last_hidden
            else:
                # For historical timesteps, compress to a single cognitive token and detach
                cog_token_t = self._extract_cog_token(last_hidden, attention_mask_t, num_patch)  # [B, 1, D]
                cog_token_t = cog_token_t.detach()
            
            cog_tokens.append(cog_token_t)

        # Stack cognitive tokens: [B, T, D]
        cog_tokens_stacked = torch.cat(cog_tokens, dim=1)  # [B, T, D]
        # import pdb; pdb.set_trace()
        
        # Pass through CausalContextTransformer (computes loss internally)
        context_token, loss, logits = self.causal_ctx_transformer(cog_tokens_stacked, labels)  # [B, 1, D], loss, [B, num_action_tokens, vocab_size]
        
        # If loss is None, set to 0
        if loss is None:
            loss = torch.tensor(0.0, device=input_ids.device)
        
        # Return MemoryVLAOutput with loss accessible via output.loss
        return MemoryVLAOutput(
            loss=loss,
            context_token=context_token,
            last_vlm_output=output_t,
            logits=logits,
        )

    @torch.inference_mode()
    def predict_action(
        self,
        image: Image,
        instruction: str,
        unnorm_key: Optional[str] = None,
        reset_context: bool = False,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Core function for VLA inference with memory. Maps input image and task instruction
        to continuous action, maintaining a buffer of past cognitive tokens.

        Args:
            image:          PIL Image as [height, width, 3]
            instruction:    Task instruction string
            unnorm_key:     Optional dataset name for un-normalization statistics
            reset_context:  If True, reset the cognitive token buffer (new episode)
            **kwargs:       Additional generation parameters

        Returns:
            Unnormalized continuous action vector (end-effector deltas)
        """
        # Reset buffer if requested (new episode)
        if reset_context:
            self._infer_cog_buffer = []
        
        image_transform = self.vision_backbone.image_transform
        tokenizer = self.llm_backbone.tokenizer
        
        # Build VLA Prompt
        prompt_builder = self.get_prompt_builder()
        prompt_builder.add_turn(
            role="human",
            message=f"What action should the robot take to {instruction.lower()}?"
        )
        prompt_text = prompt_builder.get_prompt()
        
        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            # Insert special empty token if needed
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.tensor([[29871]], device=input_ids.device)), dim=1
                )
        
        # Preprocess Image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")
        
        # Run VLM forward to get cognitive token
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            vlm_output = super().forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # Extract cognitive token from hidden states
        last_hidden = vlm_output.hidden_states[-1]  # [1, seq_len, D]
        attention_mask = torch.ones(input_ids.shape, device=input_ids.device)
        num_patch = self._get_num_patches()
        cog_token = self._extract_cog_token(last_hidden, attention_mask, num_patch)  # [1, 1, D]
        
        # Add to buffer (detached)
        self._infer_cog_buffer.append(cog_token.detach())
        
        # Keep only the last context_length tokens
        if len(self._infer_cog_buffer) > self.context_length:
            self._infer_cog_buffer = self._infer_cog_buffer[-self.context_length:]
        
        # Stack cognitive tokens from buffer
        cog_tokens_stacked = torch.cat(self._infer_cog_buffer, dim=1)  # [1, T', D]
        
        # Pass through CausalContextTransformer (no labels during inference)
        context_token, _, __ = self.causal_ctx_transformer(cog_tokens_stacked, labels=None)  # [1, 1, D]
        
        # Generate action tokens autoregressively
        if self.action_tokenizer is None:
            raise ValueError("Action tokenizer not initialized; cannot predict actions")
        
        action_dim = self.get_action_dim(unnorm_key)
        predicted_action_token_ids = []
        
        # Generate action tokens one by one
        for _ in range(action_dim):
            # Project context to vocabulary
            logits = self.causal_ctx_transformer.output_proj(context_token)  # [1, 1, vocab_size]
            
            # Take argmax to get predicted token
            next_token_id = logits.argmax(dim=-1)  # [1, 1]
            predicted_action_token_ids.append(next_token_id.item())
            
            # Update context for next prediction (append embedding of predicted token)
            if _ < action_dim - 1:  # Don't need to update after last token
                next_token_embed = self.causal_ctx_transformer._embedding_layer()(next_token_id)  # [1, 1, D]
                # For simplicity, just use the last hidden state
                # In a full autoregressive setup, we'd concat and reprocess
                context_token = next_token_embed
        
        predicted_action_token_ids = np.array(predicted_action_token_ids)
        
        # Convert token IDs to action token IDs (need to map to correct range)
        # The action tokens are the last n_bins tokens of the vocabulary
        predicted_action_token_ids = (
            self.action_tokenizer.tokenizer.vocab_size - 
            (self.action_tokenizer.n_bins - predicted_action_token_ids)
        )
        
        # Decode to normalized continuous actions
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(
            predicted_action_token_ids
        )
        
        # Un-normalize actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high = np.array(action_norm_stats["q99"])
        action_low = np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: str) -> str:
        """Validate and return the unnorm_key for accessing normalization statistics."""
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Model was trained on multiple datasets, please pass an `unnorm_key` from: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))
        
        assert unnorm_key in norm_stats, (
            f"Invalid `unnorm_key`: {unnorm_key}. Available keys: {norm_stats.keys()}"
        )
        
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict:
        """Get action normalization statistics."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return FSDP wrapping policy for distributed training."""
        vlm_wrap_policy = partial(
            _module_wrap_policy,
            module_classes={
                CausalContextTransformer,
            },
        )
        
        # Get the wrapped policies from the vision and LLM backbones
        vision_wrap_policy = self.vision_backbone.get_fsdp_wrapping_policy()
        llm_wrap_policy = self.llm_backbone.get_fsdp_wrapping_policy()
        
        # Combine all policies
        return partial(
            _or_policy,
            policies=[
                vlm_wrap_policy,
                vision_wrap_policy,
                llm_wrap_policy,
            ],
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        norm_stats: Optional[Dict] = None,
        action_tokenizer: Optional[ActionTokenizer] = None,
        context_length: int = 16,
        input_seq_len: int = 256,
        causal_num_heads: int = 8,
        causal_num_layers: int = 4,
        causal_dropout: float = 0.0,
        freeze_weights: bool = True,
        **kwargs,
    ) -> MemoryVLA_V2:
        """
        Initialize a MemoryVLA_V2 from a pretrained checkpoint.
        
        Args:
            pretrained_checkpoint: Path to checkpoint directory or .pt file
            ... (other args same as __init__)
            freeze_weights: If True, freeze all parameters after loading
            
        Returns:
            Initialized MemoryVLA_V2 model
        """
        # Initialize model
        vlm = cls(
            model_id=model_id,
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            norm_stats=norm_stats,
            action_tokenizer=action_tokenizer,
            context_length=context_length,
            input_seq_len=input_seq_len,
            causal_num_heads=causal_num_heads,
            causal_num_layers=causal_num_layers,
            causal_dropout=causal_dropout,
            **kwargs,
        )
        
        # Load checkpoint
        if pretrained_checkpoint.is_dir():
            checkpoint_pt = pretrained_checkpoint / "model.pt"
        else:
            checkpoint_pt = pretrained_checkpoint
        
        if checkpoint_pt.exists():
            state_dict = torch.load(checkpoint_pt, map_location="cpu")
            vlm.load_state_dict(state_dict, strict=False)
            overwatch.info(f"Loaded MemoryVLA_V2 checkpoint from {checkpoint_pt}")
        else:
            overwatch.warning(f"Checkpoint {checkpoint_pt} not found, using random initialization")
        
        # Freeze weights if requested
        if freeze_weights:
            for param in vlm.parameters():
                param.requires_grad = False
        
        return vlm
