from dataclasses import dataclass
from typing import Any, Dict, Type

import numpy as np
import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.proprio_tokenizer import ProprioTokenizer

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


@dataclass
class HDF5BatchTransform:
    action_tokenizer: ActionTokenizer
    proprio_tokenizer: ProprioTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    action_chunk_size: int = 1

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name = rlds_batch["dataset_name"]
        
        # For future action predictions (remove batch dimension in either case)
        if rlds_batch["action"].shape[1] > 1:  # (1, T>1, action_dim)
            action = rlds_batch["action"][0][-self.action_chunk_size:]  # (action_chunk_size, action_dim)
        else:
            action = rlds_batch["action"][0, 0]  # (action_dim,)

        # We ignore the frames for "future" actions
        num_frames = rlds_batch["observation"]["image_primary"].shape[1] - self.action_chunk_size + 1

        # Extract image data with time dimension, remove batch dimension
        img_data = rlds_batch["observation"]["image_primary"][0][:num_frames]  # [T, H, W, C]
        
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # Construct Chat-based Prompt (same for all frames, without action tokens)
        prompt_builder = self.prompt_builder_fn("openvla")
        
        # Check if proprio is available
        if "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"][0][-1]  # Take last proprio
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}? Robot current state is {self.proprio_tokenizer(proprio)}."},
                {"from": "gpt", "value": ""},
            ]
        else:
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": ""},
            ]
        
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        # labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        # Create lists of pixel_values, input_ids_list, and labels_list for each frame
        pixel_values_list = []
        input_ids_list = []
        labels_list = []
        
        for t in range(num_frames):
            # Extract frame and convert to PIL Image
            img_frame = Image.fromarray(img_data[t].astype(np.uint8))
            pixel_values = self.image_transform(img_frame)
            pixel_values_list.append(pixel_values)
            
            # Create copies of input_ids and labels for this frame
            input_ids_t = torch.tensor(input_ids)
            # labels_t = torch.tensor(labels)
            input_ids_list.append(input_ids_t)

        for t in range(self.action_chunk_size):
            action_str = self.action_tokenizer(action[t])
            action_token_ids = self.base_tokenizer(action_str, add_special_tokens=False).input_ids # e.g. [29871, 31889, 31891, 31918, 31889, 31900, 31872, 31744]
            labels_t = torch.tensor(action_token_ids) # shape: (8,)

            # labels_t[:-len(action[t])] = IGNORE_INDEX # ignore all but action tokens
            labels_t = labels_t[-len(action[t]):]  # keep only action tokens (usually removes any special tokens at the beginning)

            # # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
            # # Mask prompt tokens
            # labels_t[:] = IGNORE_INDEX
            
            # if not self.predict_stop_token:
            #     labels_t[-1] = IGNORE_INDEX

            labels_list.append(labels_t)

        # Add future actions to batch
        action_mask = None
        action = torch.tensor(action, dtype=torch.float32)
        if rlds_batch["action"].shape[1] > 1:
            if "action_mask" in rlds_batch:
                action_mask = torch.tensor(rlds_batch["action_mask"], dtype=torch.bool)

        timesteps = rlds_batch['observation']['timestep']

        # If single frame, return as tensors for backward compatibility; otherwise stack lists into tensors
        if num_frames == 1:
            pixel_values = pixel_values_list[0]
            input_ids = input_ids_list[0]
            labels = labels_list[0]
        else:
            pv_keys = pixel_values_list[0].keys()
            pixel_values = {k: torch.stack([pv[k] for pv in pixel_values_list]) for k in pv_keys}
            input_ids = torch.stack(input_ids_list)
            labels = torch.stack(labels_list)

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            actions=action,
            action_masks=action_mask,
            timesteps=timesteps,
            episode_ids=None,
        )