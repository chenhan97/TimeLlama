from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import copy
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
import warnings
#from fastchat.train.llama_flash_attn_monkey_patch import (
#    #replace_llama_attn_with_flash_attn,
#)

#replace_llama_attn_with_flash_attn()

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

#add your own token here
access_token = ""

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-13b-chat-hf")
    flash_attn: bool = False


@dataclass
class DataArguments:
    data_path: str = field(
        default="dataset/train_dataset.json", metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    output_dir: str = field(default="model")
    evaluation_strategy: str = field(default="epoch")
    #save_strategy: str = field(default="epoch")
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=1400)
    save_total_limit: int = field(default=3)
    #deepspeed: str = field(default="deepspeed_config.json")
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=1e-3)
    lr_scheduler_type: str = field(default="linear")
    gradient_accumulation_steps: int = field(default=1)
    #load_best_model_at_end: bool = field(default=True)
    # model_max_length: int = field(
    #    default=512,
    #    metadata={
    #        "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
    #    },
    # )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


class InstructionDataset(Dataset):
    def __init__(self, data_path, tokenizer, partition="train", max_words=1024):
        self.ann = json.load(open(data_path))
        if partition == "train":
            self.ann = self.ann[:]
        else:
            self.ann = self.ann[:100]

        self.max_words = max_words
        # tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer = tokenizer
        # self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        example = prompt + ann["output"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)

        example = torch.tensor(example, dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        InstructionDataset
    )
    rank0_print("Loading data...")
    train_dataset = dataset_cls(data_args.data_path, tokenizer=tokenizer)
    eval_dataset = dataset_cls(data_args.data_path, tokenizer=tokenizer, partition="eval")
    #eval_dataset = dataset_cls("dataset/eval_dataset.json", tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        token=access_token
    )
    model.config.use_cache = False
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=access_token
    )
    tokenizer.add_special_tokens(
        {

            "pad_token": "<PAD>",
        }
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    
    #if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #    trainer.train(resume_from_checkpoint=False)
    #else:
    trainer.train()
    model.config.use_cache = True
    trainer.save_state()
    trainer_save_model_safe(trainer)
    #safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
