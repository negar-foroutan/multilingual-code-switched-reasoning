# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import logging
import os
import sys
import random
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, TypeVar

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm, trange

from data_processing import load_pretraining_dataset, MultiDatasetDataloader, get_single_dataloader
from transformers import (
    AdamW,
    AutoModelForMaskedLM,
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    default_data_collator,
    DataCollatorWithPadding,
)

MODEL_PATH = {
    "mbert": "bert-base-multilingual-cased",
    "xlm-r": "xlm-roberta-base",
    "xlm-r-large": "xlm-roberta-large",
}

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(args):
    """Set seed for reproducibility."""
    if args.rand_seed:
        args.seed = np.random.randint(100)
        logger.info(f"New seed: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


@dataclass
class PretrainingArguments:
    """ Arguments for training cross-lingual Query matrix. """

    mono_alpha: float = field(
        default=1.0,
        metadata={"help": "Attention ratio for monolingual mask; meaning that `(1-mono_alpha)` of monolingual attentions for original Query (Q) are masked."}
    )
    cross_alpha: float = field(
        default=0.3, 
        metadata={"help": "Attention ratio for cross-lingual mask; meaning that (1-cross_alpha)` of cross-lingual attentions for Q_cross are masked."}
    )
    mono_eval_alpha: float = field(
        default=1.0, 
        metadata={"help": "Inference mode: Attention ratio for monolingual mask; meaning that `(1-mono_alpha)` of monolingual attentions for original Query (Q) are masked."}
    )
    cross_eval_alpha: float = field(
        default=0.0, 
        metadata={"help": "Inference mode: Attention ratio for cross-lingual mask; meaning that (1-cross_alpha)` of cross-lingual attentions for Q_cross are masked."}
    )
    freeze_the_rest: bool = field(
        default=True, 
        metadata={"help": "Whether freeze the rest of the parameters (apart from the new cross-lingual Query matrix) or not."}
    )
    train_data_percentage: float = field(
        default=1.0, 
        metadata={"help": "Percentage of training data to use."}
    )
    temperature: float = field(
        default=5.0, 
        metadata={"help": "Temperature for temperature sampling."}
    )
    sampling_strategy: str = field(
        default="size_proportional",
        metadata={
            "help": "The sampling strategy to use for training."},
    )
    code_switched_format: str = field(
        default="en-x",
        choices=['en-x', 'ex-en', 'x-x'],
        metadata={
            "help": "The format of code-switched data (en-x, x-en, or x-x)."},
    )
    model_type: str = field(
        default="bert", metadata={"help": "Model type"}
    )
    mlm: bool = field(
        default=True, metadata={"help": "Whether uses MLM objective."}
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    block_size: int = field(
        default=512, metadata={"help": "Optional input sequence length after tokenization."}
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."}
    )
    max_grad_norm: float = field(
        default=1.0, metadata={"help": "Max gradient norm."})
    output_dir: str = field(
        default=None, metadata={"help": "Output directory path."}
    )
    log_dir: str = field(
        default=None, metadata={"help": "Log directory path."}
    )
    overwrite_output_dir: bool = field(
        default=True, metadata={"help": "Whether overwrite the output dir."}
    )
    data_language: str = field(
        default=None, metadata={"help": "Data language."}
    )
    data_language_pairs: str = field(
        default=None, metadata={"help": "Data language pairs."}
    )

    save_steps: int = field(
        default=10000, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(
        default=3000, metadata={"help": "Log every X updates steps."}
    )
    evaluate_during_training: bool = field(
        default=True, metadata={"help": "Whether to evaluate during training or not."}
    )
    weight_init: str = field(
        default="pre", metadata={"help": "Initial weights."}
    )
    rand_seed: bool = field(
        default=False,  metadata={"help": "Whether set a seed or not."}
    )
    do_train: bool = field(
        default=False, metadata={"help": "Whether to run training."}
    )
    do_eval: bool = field(
        default=True, metadata={"help": "Whether to run eval on the dev set."}
    )
    do_predict: bool = field(
        default=False, metadata={"help": "Whether to run predictions on the test set."}
    )
    per_device_train_batch_size: int = field(
        default=32, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=32, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    num_train_epochs: float = field(
        default=3.0, metadata={"help": "Total number of training epochs to perform."}
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Linear warmup over warmup_steps."}
    )
    learning_rate: float = field(
        default=2e-5, metadata={"help": "The initial learning rate for AdamW."}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."}
        )
    adam_beta1: float = field(
        default=0.9, metadata={"help": "Beta1 for AdamW optimizer"}
    )
    adam_beta2: float = field(
        default=0.999, metadata={"help": "Beta2 for AdamW optimizer"}
    )
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."}
    )
    local_rank: int = field(
        default=-1, metadata={"help": "For distributed training: local_rank"}
    )
    no_cuda: bool = field(
        default=False, metadata={"help": "Do not use CUDA even when it is available"}
    )
    seed: int = field(
        default=65, metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    model_name_or_path: str = field(
        default="bert-base-multilingual-cased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    task_name: Optional[str] = field(
        default="mlm", metadata={"help": "The name of the task."}
    )
    dataset_name: Optional[str] = field(
        default="xnli", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    val_data_ratio: Optional[float] = field(
        default=0.05,
        metadata={"help": "The ratio of validation data from the training data."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If set, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    fp16: bool = field(
        default=False,
        metadata={"help":"Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"})

    fp16_opt_level: str = field(
        default="O1",
        metadata={"help":"For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html"})

    server_ip: str = field(default="", metadata={ "help": "For distant debugging."})
    server_port: str = field(default="", metadata={ "help": "For distant debugging."})


class SelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.cross_lingual_query = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        mixed_cross_lingual_query_layer = self.cross_lingual_query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        cross_lingual_query_layer = self.transpose_for_scores(mixed_cross_lingual_query_layer)
        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        cross_lingual_attention_scores = torch.matmul(cross_lingual_query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1,
                                              dtype=torch.long,
                                              device=hidden_states.device).view(-1, 1)
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        chunk_size = attention_mask.shape[0] // 2
        attention_splits = attention_mask.split(chunk_size, dim=0)
        mono_attention_mask = attention_splits[0]
        cross_lingual_attention_mask = attention_splits[1]

        attention_scores = attention_scores + mono_attention_mask

        cross_lingual_attention_scores = cross_lingual_attention_scores / math.sqrt(self.attention_head_size)
        cross_lingual_attention_scores = cross_lingual_attention_scores + cross_lingual_attention_mask

        max_value = torch.maximum(attention_scores.max(dim=-1, keepdim=True)[0],
                                  cross_lingual_attention_scores.max(dim=-1, keepdim=True)[0]).detach()
        attention_scores = attention_scores - max_value
        cross_lingual_attention_scores = cross_lingual_attention_scores - max_value

        attention_probs = torch.exp(attention_scores) + torch.exp(cross_lingual_attention_scores)

        attention_probs = attention_probs / (attention_probs.sum(dim=-1, keepdim=True) + 1e-10)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


def set_attention_module(model, config, args):
    """ Set the cross-lingual attention module for each layer of the model. """
    for layer in range(config.num_hidden_layers):
        att_module = SelfAttention(config)
        # copy the weights from the pretrained model
        if args.model_type == "mbert":
            orig_att_module = model.bert.encoder.layer[layer].attention.self
        elif args.model_type == "xlm-r":
            orig_att_module = model.roberta.encoder.layer[layer].attention.self
        else:
            raise NotImplementedError(f"invalid model_type of {args.model_type} is not supported")

        orig_state_dicts = orig_att_module.state_dict()

        with torch.no_grad():
            orig_state_dicts["cross_lingual_query.weight"] = orig_state_dicts["query.weight"].clone()
            orig_state_dicts["cross_lingual_query.bias"] = orig_state_dicts["query.bias"].clone()

            att_module.load_state_dict(orig_state_dicts)

            if args.model_type == "mbert":
                model.bert.encoder.layer[layer].attention.self = att_module
            elif args.model_type == "xlm-r":
                model.roberta.encoder.layer[layer].attention.self = att_module


def remove_unneeded_padding(input_ids_, attention_mask_, token_type_ids_):
    """ Remove unneeded padding for the input_ids, attention_mask, and token_type_ids. """
    max_seq_length = int(attention_mask_[:, 0, 0, :].sum(dim=1).max().item())

    input_ids_output = input_ids_[:, :max_seq_length]
    attention_mask_output = attention_mask_[:, :, :max_seq_length, :max_seq_length]
    token_type_ids_output = token_type_ids_[:, :max_seq_length]
    return input_ids_output, attention_mask_output, token_type_ids_output


def mask_tokens(inputs: torch.Tensor, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()

    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(args, train_dataset, model, tokenizer, eval_dataset, data_collator):
    """ Train the model """
    record_result = []

    if args.local_rank in [-1, 0]:
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        args.log_dir = os.path.join(args.log_dir,
                                    current_time + "_" + args.output_dir.split("/")[-1] + "_seed_" + str(args.seed))
        tb_writer = SummaryWriter(log_dir=args.log_dir)

    args.train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)

    train_dataloader = MultiDatasetDataloader(
        {
            f'{lang_name}': get_single_dataloader(lang_name, dataset, args.train_batch_size, data_collator, args)
            for lang_name, dataset in train_dataset.items()
        },
        sampling_strategy=args.sampling_strategy,
        temperature=args.temperature,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", sum([len(train_dataset[l]) for l in train_dataset]))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_device_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    for _ in train_iterator:

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            batch = {t: batch[t].to(args.device) for t in batch}
            model.train()

            input_ids_, attention_mask_, token_type_ids_ = remove_unneeded_padding(batch["input_ids"],
                                                                                   batch["attention_mask"],
                                                                                   batch["token_type_ids"])
            mono_attention_mask = attention_mask_[:, 0, :, :]
            cross_attention_mask = attention_mask_[:, 1, :, :]
            attention_mask_ = torch.cat((mono_attention_mask, cross_attention_mask), dim=0)

            input_ids_, labels = mask_tokens(input_ids_.long().to("cpu"), tokenizer, args)

            outputs = model(input_ids=input_ids_.long().to(args.device), attention_mask=attention_mask_.to(args.device),
                            token_type_ids=token_type_ids_.to(args.device), labels=labels.to(args.device))

            loss = outputs.loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.detach().item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0 or global_step == args.save_steps:
                    logs = {}
                    # Log metrics
                    if (args.local_rank == -1 and args.evaluate_during_training):  # Only evaluate when single GPU otherwise metrics may not average well

                        results = evaluate(args, model, tokenizer, eval_dataset, data_collator)
                        record_result.append(results)
                        for key, value in results.items():
                            eval_key = "eval/{}".format(key)
                            logs[eval_key] = value
                        print(results)

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["train/loss"] = loss_scalar
                    logs["train/perplexity"] = torch.exp(torch.tensor(loss_scalar)).detach().item()
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    torch.save({i[0]: i[1] for i in model.state_dict().items() if "cross_lingual" in i[0]},
                                   os.path.join(output_dir, "query.pt"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    torch.save(record_result, os.path.join(args.output_dir, "result.pt"))

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, eval_dataset, data_collator, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    top1 = AverageMeter()

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_device_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_dataloader = MultiDatasetDataloader(
        {
            f'{lang_name}': get_single_dataloader(lang_name, dataset, args.eval_batch_size, data_collator, args)
            for lang_name, dataset in eval_dataset.items()
        },
        evaluation=True
    )

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        with torch.no_grad():

            inputs = {t: batch[t].to(args.device) for t in batch}

            input_ids_, attention_mask_, token_type_ids_ = remove_unneeded_padding(inputs["input_ids"],
                                                                                   inputs["attention_mask"],
                                                                                   inputs["token_type_ids"])
            mono_attention_mask = attention_mask_[:, 0, :, :]
            cross_attention_mask = attention_mask_[:, 1, :, :]
            attention_mask_ = torch.cat((mono_attention_mask, cross_attention_mask), dim=0)

            input_ids_, labels = mask_tokens(input_ids_.long().to("cpu"), tokenizer, args)
            labels = labels.to(args.device)

            outputs = model(input_ids=input_ids_.long().to(args.device), attention_mask=attention_mask_.to(args.device),
                            token_type_ids=token_type_ids_.to(args.device), labels=labels.to(args.device))

            lm_loss = outputs["loss"]
            prediction_scores = outputs["logits"].detach()

            vocab_size = prediction_scores.size(-1)
            acc = accuracy(prediction_scores.view(-1, vocab_size).data, labels.view(-1))[0]
            top1.update(acc.item(), labels.view(-1).size(0))

            eval_loss += lm_loss.detach().mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"loss": eval_loss, "perplexity": perplexity, 'acc': top1.avg}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    with open(os.path.join(args.output_dir, "logs_eval_results.txt"), "a") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def accuracy(output_orig, label, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    index = torch.nonzero(label+100)
    target = label[index].view(-1)
    output = output_orig[index].squeeze(1)

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def find_last_checkpoint(directory):
    checkpoint_folders = [f for f in os.listdir(directory) if f.startswith('checkpoint')]
    if checkpoint_folders:
        # sort the checkpoint folders by the step in their name
        checkpoint_folders.sort(key=lambda x: int(x.split('_')[-1]))
        return os.path.join(directory, checkpoint_folders[-1])
    else:
        return None


def main():
    parser = HfArgumentParser(PretrainingArguments)

    if sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]
   
    args.data_language_pairs = args.data_language_pairs.split(";")
    args.model_name_or_path = MODEL_PATH[args.model_type]
    args.config_name = args.model_name_or_path

    logger.info("*************** Cross-Lingual Query Pretraining Step *************************")
    logger.info(f"*Data Language Pairs: {args.data_language_pairs}.\n")
    logger.info(f"*Output dir: {args.output_dir}\n")
    logger.info(f"*Dataset Name: {args.dataset_name}\n")
    logger.info("**************************************************************")

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab


    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir
    )
    config.data_language_pairs = args.data_language_pairs

    tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            cache_dir=args.cache_dir)

    if os.path.exists(args.model_name_or_path):
        logger.info(f"Loading model from {args.model_name_or_path}")
        model = torch.load(os.path.join(args.model_name_or_path, "query.pt"))
    else:
        logger.info(f"Loading model from {args.model_name_or_path}")
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )

        set_attention_module(model, config, args)

        if args.freeze_the_rest:
            for name, param in model.named_parameters():
                if "cross_lingual_query" not in name:
                    param.requires_grad = False      

    args.num_hidden_layers = config.num_hidden_layers

    model.to(args.device)

    if args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    ###

    if args.pad_to_max_length:
        data_collator = default_data_collator
    elif args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Load the dataset
    train_dataset = {}
    eval_dataset = {}

    for language_pair in args.data_language_pairs:
        dataset = load_pretraining_dataset(language_pair, args.dataset_name, tokenizer, args, padding)
        train_dataset[language_pair] = dataset["train"]
        eval_dataset[language_pair] = dataset["eval"]
        logger.info("********************************************************")

    print("Train data size: {}".format(sum([len(train_dataset[language_pair]) for language_pair in train_dataset])))
    print("Test data size: {}".format(sum([len(eval_dataset[language_pair]) for language_pair in eval_dataset])))

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, eval_dataset, data_collator)
        results = evaluate(args, model, tokenizer, eval_dataset, data_collator)
        print(f"Final results: {results}")
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    elif args.do_eval:
        results = evaluate(args, model, tokenizer, eval_dataset, data_collator)
        print(f"FinEvaluational results: {results}")


if __name__ == "__main__":
    main()