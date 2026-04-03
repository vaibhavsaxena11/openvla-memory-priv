"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSBatchTransform, RLDSDataset


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    data_format: str = "rlds",
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    **kwargs
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
    action_tokenizer = ActionTokenizer(tokenizer)
    # batch_transform = RLDSBatchTransform(
    #     action_tokenizer, tokenizer, image_transform, prompt_builder_fn, predict_stop_token=predict_stop_token
    # )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )

    if data_format == "rlds":
        batch_transform = RLDSBatchTransform(
            action_tokenizer, tokenizer, image_transform, prompt_builder_fn, predict_stop_token=predict_stop_token
        )
        # Build RLDS Iterable Dataset
        cls = RLDSDataset if not episodic else EpisodicRLDSDataset
        dataset = cls(
            data_root_dir,
            data_mix,
            batch_transform,
            resize_resolution=default_image_resolution[1:],
            shuffle_buffer_size=shuffle_buffer_size,
            train=train,
            image_aug=image_aug,
        )
    elif data_format == "hdf5":
        from prismatic.vla.proprio_tokenizer import ProprioTokenizer
        # from rldb.vla.datasets import HDF5Dataset, HDF5BatchTransform

        from rldb.vla.datasets import HDF5Dataset
        from prismatic.vla.datasets import HDF5BatchTransform # modified for MemoryVLA_V2

        proprio_tokenizer = ProprioTokenizer(tokenizer, bins=256, token_begin_idx=31000)
        batch_transform = HDF5BatchTransform(
            action_tokenizer, proprio_tokenizer, tokenizer, image_transform, prompt_builder_fn, predict_stop_token=predict_stop_token
        )

        # Extract configurable parameters from kwargs
        action_keys = kwargs.get("action_keys", ("actions_abs",))
        frame_stack = kwargs.get("frame_stack", 1)
        cls = HDF5Dataset
        dataset = cls(
            data_root_dir,
            data_mix,
            batch_transform,
            # resize_resolution=default_image_resolution[1:], # TODO add support in HDF5Dataset
            # shuffle_buffer_size=shuffle_buffer_size, # TODO add support in HDF5Dataset
            # image_aug=image_aug, # TODO add support in HDF5Dataset
            frame_stack=frame_stack,
            seq_length=1,
            # action_config={"actions_abs": {"normalization": "gaussian"}},
            load_next_obs=False,
            # action_keys=("actions",),
            # dataset_keys=("actions", "rewards", "dones"),
            action_keys=action_keys,
            dataset_keys=action_keys,
            obs_keys=("agentview_image",), # TODO somehow move to vla config
            # obs_keys=("agentview_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",), # TODO somehow move to vla config
            hdf5_cache_mode=None,
            # ds_weights=None, # [1.0, 2.0, 1.0],  # Relative sampling weights # TODO somehow move to vla config
            # ds_langs=["pick up the object", "place the object", "push the object"],
            # ds_langs=["put the bowl in the basket", "put the bowl in the basket", "put the bowl in the basket", "put the bowl in the basket"],  # Per-dataset languages # TODO somehow move to vla config
            # demo_limit=None,  # Limit demos per dataset (None for no limit)
            train=train,
            normalize_weights_by_ds_size=True,
        )
    else:
        raise ValueError(f"Unsupported data_format: {data_format}")

    return dataset, action_tokenizer, collator
