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
""" Finetuning the library models for sequence classification on XNLI (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import json
import logging
import os
import copy

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import time
import math
import sys
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm, trange

from data_processing import RuleTakerDatasetSepAttentionMask, LeapOfThoughtDatasetSepAttentionMask, MultiDatasetDataloader, get_single_dataloader

from dataclasses import dataclass, field
from typing import Optional
import evaluate as evaluate_metrics

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification, 
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    default_data_collator,
    get_linear_schedule_with_warmup,
    AdamW,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATASET_CLASSES = {"ruletaker": RuleTakerDatasetSepAttentionMask, "lot": LeapOfThoughtDatasetSepAttentionMask}
              
MODEL_PATH = {
    "mbert": "bert-base-multilingual-cased",
    "xlm-r": "xlm-roberta-base", 
    "xlm-r-large": "xlm-roberta-large", 
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "xlm-tlm": "xlm-mlm-tlm-xnli15-1024",
    "xlm": "xlm-mlm-xnli15-1024",
}

@dataclass
class ReasoningTrainingArguments:
    """ Arguments for Reasoning training. """

    load_from_checkpoint: bool = field(
        default=False, 
        metadata={"help": "Whether to load from a checkpoint."}
    )
    checkpoint_model_path : Optional[str] = field(
        default=None, 
        metadata={"help": "Path to checkpoint model."}
    )
    load_query_pretrained: bool = field(
        default=False, 
        metadata={"help": "Whether to load a pretrained query model."}
    )
    query_pretrained_path: Optional[str] = field(
        default=None, 
        metadata={"help": "Path to pretrained query model."}
    )
    bitfit: bool = field(
        default=False,
        metadata={"help": "Whether to use bitfit."}
    )
    num_labels: int = field(
        default=2,
        metadata={"help": "Number of labels to use."}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Warmup ratio."}
    )
    mono_alpha: float = field(
        default=1.0,
        metadata={"help": "Attention ratio for monolingual mask; meaning that `(1-mono_alpha)` of monolingual attentions for original Query (Q) are masked."}
    )
    cross_alpha: float = field(
        default=0.3,
        metadata={"help": "Attention ratio for cross-lingual mask; meaning that (1-cross_alpha)` of cross-lingual attentions for Q_cross are masked."}
    )
    cross_mono_alpha: float = field(
        default=0.3,
        metadata={"help": "Attention ratio for cross-lingual mask; meaning that (1-cross_alpha)` of monolingual attentions for Q_cross are masked."}
    )
    mono_eval_alpha: float = field(
        default=1.0,
        metadata={"help": "Inference mode: Attention ratio for monolingual mask; meaning that `(1-mono_alpha)` of monolingual attentions for original Query (Q) are masked."}
    )
    cross_eval_alpha: float = field(
        default=0.0,
        metadata={"help": "Inference mode: Attention ratio for cross-lingual mask; meaning that (1-cross_alpha)` of cross-lingual attentions for Q_cross are masked."}
    )
    cross_mono_eval_alpha: float = field(
        default=0.0,
        metadata={"help": "Inference mode: Attention ratio for cross-lingual mask; meaning that (1-cross_alpha)` of monolingual attentions for Q_cross are masked."}
    )
    model_type: str = field(
        default="mbert",
        metadata={
            "help": "Model type selected in the list: " + ", ".join(MODEL_PATH.keys())
        },
    )
    language1: str = field(default="", metadata={"help": "Language 1"})
    language2: str = field(default="", metadata={"help": "Language 2"})
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    data_base_dir: Optional[str] = field(metadata={"help": "The base directory of the data."}
    )
    train_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the training data."},
    )
    val_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the validation data."},
    )
    test_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the test data."},
    )
    rule_taker_depth_level: Optional[str] = field(
        default=None,
        metadata={"help": "The depth level of the ruletaker dataset."},
    )
    transformer_classifier_layers: int = field(
        default=1,
        metadata={
            "help": "Number of layers for the transformer classifier on top of the MultiLM"
        },
    )
    classifier_is_bert: bool = field(
        default=True,
        metadata={"help": "If True, the classifier will be a randomly initialized BERT-like model. "
                          "if False, classifier will be a pretrained ALBERT-base model"}
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )
    max_steps: int = field(
        default=-1,
        metadata={
            "help": "If > 0: set total number of training steps to perform. Override num_train_epochs."
        },
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Max gradient norm."}
    )
    output_dir: str = field(
        default=None,
        metadata={"help": "Output directory path."}
    )
    base_dir: str = field(
        default=None,
        metadata={"help": "Base directory path."}
    )
    logging_dir: str = field(
        default=None,
        metadata={"help": "Log directory path."}
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={"help": "Whether overwrite the output dir."}
    )
    data_language: str = field(default=None, metadata={"help": "Data language."})
    model_language: str = field(
        default=None, metadata={"help": "Model language or type."}
    )
    save_steps: int = field(
        default=36813, metadata={"help": "Save checkpoint every X updates steps."}
    )
    logging_steps: int = field(
        default=3000, metadata={"help": "Log every X updates steps."}
    )
    evaluate_during_training: bool = field(
        default=True, metadata={"help": "Whether to evaluate during training or not."}
    )
    weight_pertub: bool = field(
        default=False, metadata={"help": "Whethre purturb weights or not."}
    )
    weight_init: str = field(
        default="pre", metadata={"help": "Initial weights."}
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Linear warmup over warmup_steps."}
    )

    set_seed: bool = field(
        default=True, metadata={"help": "Whether set a seed or not."}
    )
    rand_seed: bool = field(
        default=False, metadata={"help": "Whether set a seed or not."}
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
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
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )
    learning_rate: float = field(
        default=2e-5, metadata={"help": "The initial learning rate for AdamW."}
    )
    num_train_epochs: float = field(
        default=3.0, metadata={"help": "Total number of training epochs to perform."}
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
        default=65,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )
    model_name_or_path: str = field(
        default="bert-base-multilingual-cased",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    task_name: Optional[str] = field(
        default="xnli", metadata={"help": "The name of the task (ner, pos...)."}
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    dataset_name: Optional[str] = field(
        default="xnli",
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a csv or JSON file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to predict on (a csv or JSON file)."
        },
    )
    text_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The column name of text to input in the file (a csv or JSON file)."
        },
    )
    label_column_name: Optional[str] = field(
        default="label",
        metadata={
            "help": "The column name of label to input in the file (a csv or JSON file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If set, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
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
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    sampling_strategy: str = field(
        default="size_proportional",
        metadata={
            "help": "The sampling strategy to use for training."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={
            "help": "Whether to return all the entity levels during evaluation or just the overall ones."
        },
    )
    fp16: bool = field(
        default=False,
        metadata={
            "help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"
        },
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
            "See details at https://nvidia.github.io/apex/amp.html"
        },
    )
    server_ip: str = field(default="", metadata={"help": "For distant debugging."})
    server_port: str = field(default="", metadata={"help": "For distant debugging."})

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()


def set_seed(args):
    if args.rand_seed:
        args.seed = np.random.randint(100)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def freeze_parameters(model, args):
    """ Freezes the parameters of the model for Bitfit or Adapter settings. """
    if args.bitfit and args.is_adapter:
        Exception("Bitfit and Adapter cannot be used together")
        
    if args.bitfit:
        for name, param in model.named_parameters():
            if ".bias" in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif args.is_adapter:
        for name, param in model.named_parameters():
            if "adapter" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    if args.model_type == "xlm-r":
        model.classifier.dense.weight.requires_grad = True
        model.classifier.dense.bias.requires_grad = True
        model.classifier.out_proj.weight.requires_grad = True
        model.classifier.out_proj.bias.requires_grad = True
            
    if args.model_type == "mbert":
        model.bert.pooler.dense.weight.requires_grad = True
        model.bert.pooler.dense.bias.requires_grad = True
        model.classifier.weight.requires_grad = True
        model.classifier.bias.requires_grad = True


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
        ###
        self.cross_lingual_query = nn.Linear(config.hidden_size, self.all_head_size)
        ###

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
        ##
        mixed_cross_lingual_query_layer = self.cross_lingual_query(hidden_states)
        ##

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
        ###
        cross_lingual_query_layer = self.transpose_for_scores(mixed_cross_lingual_query_layer)
        ###
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
        
        ##
        cross_lingual_attention_scores = torch.matmul(cross_lingual_query_layer, key_layer.transpose(-1, -2))
        ##

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
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
    
        ##
        chunk_size = attention_mask.shape[0] // 2
        attention_splits = attention_mask.split(chunk_size, dim=0)
        mono_attention_mask = attention_splits[0]
        cross_lingual_attention_mask = attention_splits[1]
        
        attention_scores = attention_scores + mono_attention_mask
        
        if not self.is_mono:

            cross_lingual_attention_scores = cross_lingual_attention_scores / math.sqrt(self.attention_head_size)
            cross_lingual_attention_scores = cross_lingual_attention_scores + cross_lingual_attention_mask
            
            max_value = torch.maximum(attention_scores.max(dim=-1, keepdim=True)[0], cross_lingual_attention_scores.max(dim=-1, keepdim=True)[0]).detach()
            attention_scores = attention_scores - max_value
            cross_lingual_attention_scores = cross_lingual_attention_scores - max_value
            
            attention_probs = torch.exp(attention_scores) + torch.exp(cross_lingual_attention_scores)
                
            attention_probs = attention_probs / (attention_probs.sum(dim=-1, keepdim=True) + 1e-10)
            ##
        else:
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        
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
        
def remove_unneeded_padding(input_ids_, attention_mask_, token_type_ids_):
    """ Remove unneeded padding from the input ids, attention mask, and token type ids."""
    max_seq_length = int(attention_mask_[:, 0, 0, :].sum(dim=1).max().item())

    input_ids_output = input_ids_[:, :max_seq_length]
    attention_mask_output = attention_mask_[:, :, :max_seq_length, :max_seq_length]
    token_type_ids_output = token_type_ids_[:, :max_seq_length]
    return input_ids_output, attention_mask_output, token_type_ids_output


def train(args, train_dataset, eval_dataset, model, data_collator, compute_metrics, tokenizer):
    """Train the model"""
    record_result = []

    if args.local_rank in [-1, 0]:
        log_dir = f"{args.logging_dir}_data_seed_{str(args.seed)}"
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        tb_writer = SummaryWriter(log_dir=log_dir)

    args.train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)

    train_dataloader = MultiDatasetDataloader(
        {
        f'{lang_name}': get_single_dataloader(lang_name, dataset, args.train_batch_size, data_collator, args)
                for lang_name, dataset in train_dataset.items()
            },
        sampling_strategy=args.sampling_strategy,
    )

    t_total = (len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")
    ) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model.to(args.device)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_device_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],)

    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):
            
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            batch = {t: batch[t].to(args.device) for t in batch}
            # Set a flag in all attention moules
            args.is_mono = False if "-" in batch["lang_name"] else True
            for layer in range(args.num_hidden_layers):
                if isinstance(model, nn.DataParallel):
                    if args.model_type == "mbert":
                        model.module.bert.encoder.layer[layer].attention.self.is_mono = args.is_mono
                    elif args.model_type == "xlm-r":
                        model.module.roberta.encoder.layer[layer].attention.self.is_mono = args.is_mono
                else:
                    if args.model_type == "mbert":
                        model.bert.encoder.layer[layer].attention.self.is_mono = args.is_mono
                    elif args.model_type == "xlm-r":
                        model.roberta.encoder.layer[layer].attention.self.is_mono = args.is_mono
                
            model.train()
            
            input_ids_, attention_mask_, token_type_ids_ = remove_unneeded_padding(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])
            mono_attention_mask = attention_mask_[:, 0, :, :]
            cross_attention_mask = attention_mask_[:, 1, :, :]
            attention_mask_ = torch.cat((mono_attention_mask, cross_attention_mask), dim=0)
            
            outputs = model(input_ids=input_ids_.to(args.device), attention_mask=attention_mask_.to(args.device),
                            token_type_ids=token_type_ids_.to(args.device), labels=batch["labels"].to(args.device))

            loss = outputs["loss"]  # model outputs are always tuple in transformers (see doc)

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

            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                    args.gradient_accumulation_steps >= len(epoch_iterator) == (step + 1)
            ):
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                    or global_step == args.save_steps):
                    logs = {}
                    if (args.local_rank == -1 and args.evaluate_during_training):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, eval_dataset, model, data_collator, compute_metrics)
                        record_result.append(results)
                        for key, value in results.items():
                            eval_key = "eval/{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["train/loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if (
                    args.local_rank in [-1, 0]
                    and args.save_steps > 0
                    and global_step % args.save_steps == 0
                ):
                    # Save model checkpoint
                    if logs["eval/accuracy"] > 0.60:
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        tokenizer.save_pretrained(output_dir)
                        torch.save(model, os.path.join(output_dir, "model.pt"))

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    torch.save(record_result, os.path.join(args.output_dir, "result.pt"))

    return global_step, tr_loss / global_step


def evaluate(args, eval_dataset, model, data_collator, compute_metrics, prefix=""):

    results = {}
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_device_eval_batch_size * max(1, args.n_gpu)
    eval_dataloader = MultiDatasetDataloader(
        {
        f'{lang_name}': get_single_dataloader(lang_name, dataset, args.eval_batch_size, data_collator, args)
                for lang_name, dataset in eval_dataset.items()
            },
        evaluation=True
    )
    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        with torch.no_grad():
    
            inputs = {t: batch[t].to(args.device) for t in batch}
            args.is_mono = False if "-" in batch["lang_name"] else True
            for layer in range(args.num_hidden_layers):
                if isinstance(model, nn.DataParallel):
                    if args.model_type == "mbert":
                        model.module.bert.encoder.layer[layer].attention.self.is_mono = args.is_mono
                    elif args.model_type == "xlm-r":
                        model.module.roberta.encoder.layer[layer].attention.self.is_mono = args.is_mono
                else:
                    if args.model_type == "mbert":
                        model.bert.encoder.layer[layer].attention.self.is_mono = args.is_mono
                    elif args.model_type == "xlm-r":
                        model.roberta.encoder.layer[layer].attention.self.is_mono = args.is_mono
                    
            input_ids_, attention_mask_, token_type_ids_ = remove_unneeded_padding(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])
            
            mono_attention_mask = attention_mask_[:, 0, :, :]
            cross_attention_mask = attention_mask_[:, 1, :, :]
            attention_mask_ = torch.cat((mono_attention_mask, cross_attention_mask), dim=0)
            
            outputs = model(input_ids=input_ids_.to(args.device), attention_mask=attention_mask_.to(args.device),
                            token_type_ids=token_type_ids_.to(args.device), labels=inputs["labels"].to(args.device))
            ###
            logits = outputs["logits"]
            tmp_eval_loss = outputs["loss"]

            eval_loss += tmp_eval_loss.detach().mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
            )

    eval_loss = eval_loss / nb_eval_steps
    results["loss"] = eval_loss
    result = compute_metrics(EvalPrediction(predictions=preds, label_ids=out_label_ids))
    results.update(result)

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    with open(os.path.join(args.output_dir, "logs_eval_results.txt"), "a") as writer:
        for key in sorted(results.keys()):
            writer.write("%s = %s\n" % (key, str(results[key])))

    return results       
    
def main():

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    start = time.time()
    
    parser = HfArgumentParser(ReasoningTrainingArguments)

    if sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]  

    if args.language1 != "" and args.language2 != "":
        args.train_languages = [args.language1, args.language2]
        args.train_data_percentage = 0.5
    else:
        args.train_languages = [args.language1]
        args.train_data_percentage = 1

    
    if args.do_train and "ids" in args.dataset_name and not "with-token" in args.query_pretrained_path:
        ValueError("You need to use a query pretrained model with tokens for ids training")
    
    if args.do_train and not "ids" in args.dataset_name and "with-token" in args.query_pretrained_path:
        ValueError("You need to use a query pretrained model without tokens for training")
            
    args.model_name_or_path = MODEL_PATH[args.model_type]
    args.config_name = args.model_name_or_path
    args.tokenizer_name = args.model_name_or_path

    if args.rule_taker_depth_level == "5" and args.do_train:
        ValueError("5 is not a valid depth level for the training!")

    if args.rand_seed:
        args.seed = np.random.randint(100)
        logger.info(f"New seed: {args.seed}")

    if "lot" in args.dataset_name:
        args.train_data_paths = {lang: os.path.join(args.data_base_dir, lang, "randomized_hypernyms_training_mix_short_train.jsonl") for lang in args.train_languages}
        args.val_data_paths = {lang: os.path.join(args.data_base_dir, lang, "randomized_hypernyms_training_mix_short_dev.jsonl") for lang in args.train_languages}
    elif "ruletaker" in args.dataset_name:
        args.train_data_paths = {lang: os.path.join(args.data_base_dir, lang, "original", f"depth-{args.rule_taker_depth_level}", "train.jsonl",) for lang in args.train_languages}
        args.val_data_paths = {lang: os.path.join(args.data_base_dir, lang, "original", f"depth-{args.rule_taker_depth_level}", "dev.jsonl",) for lang in args.train_languages}

    label_list = [1, 0]

    # Labels
    args.num_labels = len(label_list)
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry(f"run_reasoning_{model_args.model_type}", model_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info(f"Training/evaluation parameters {args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = 0 if args.no_cuda else 1 #torch.cuda.device_count()
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
    if args.set_seed:
        set_seed(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_labels,
        cache_dir=args.cache_dir,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
        revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
    )

    if args.load_from_checkpoint:
        logger.info(f"Loading model from checkpoint : {args.checkpoint_model_path}")
        model = torch.load(args.checkpoint_model_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
            revision=args.model_revision,
            use_auth_token=True if args.use_auth_token else None,
        )
        
        if args.load_query_pretrained:
            # call a function to load the pretrained query weights
            logger.info(f"\nLoading a pretrained model for the cross-lingual query : {args.query_pretrained_path}\n")
            model_state_dict = copy.deepcopy(model.state_dict())
            query_model = torch.load(args.query_pretrained_path)
            
            for layer in range(config.num_hidden_layers):
                att_module = SelfAttention(config)
                if args.model_type == "mbert":
                    model.bert.encoder.layer[layer].attention.self = att_module
                elif args.model_type == "xlm-r":
                    model.roberta.encoder.layer[layer].attention.self = att_module
            
            model.load_state_dict(query_model, strict=False)
            model.load_state_dict(model_state_dict, strict=False)

        else:
            for layer in range(config.num_hidden_layers):
                att_module = SelfAttention(config)
                ## copy the weights from the pretrained model
                if args.model_type == "mbert":
                    bert_att_module = model.bert.encoder.layer[layer].attention.self
                elif args.model_type == "xlm-r":
                    bert_att_module = model.roberta.encoder.layer[layer].attention.self
                    
                bert_state_dicts = bert_att_module.state_dict()

                with torch.no_grad():
                    bert_state_dicts["cross_lingual_query.weight"] = bert_state_dicts["query.weight"].clone()
                        
                    bert_state_dicts["cross_lingual_query.bias"] = bert_state_dicts["query.bias"].clone()
                    att_module.load_state_dict(bert_state_dicts)
                    if args.model_type == "mbert":
                        model.bert.encoder.layer[layer].attention.self = att_module
                    elif args.model_type == "xlm-r":
                        model.roberta.encoder.layer[layer].attention.self = att_module
    
    
    if "xlm-r" in args.model_name_or_path:
        xlm_lm_model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
        pretrained_pooler_module = xlm_lm_model.lm_head.dense
        model.classifier.dense = pretrained_pooler_module
            
    # Freeze parameters if needed (Bitfit & Adapter)
    freeze_parameters(model, args)
                    
    args.num_hidden_layers = config.num_hidden_layers
    # Padding strategy
    if args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Load datasets
    if args.do_train: 
        random.seed(65)
        np.random.seed(65)
        
        train_dataset = {}
        for lang in args.train_data_paths:
            cross_alpha = args.cross_alpha
            if "-" in lang:
                mono_alpha = args.cross_mono_alpha # Cross-lingual Data
            else:
                mono_alpha = args.mono_alpha # Mono data
                
            train_dataset[lang] = DATASET_CLASSES[args.dataset_name](
                tokenizer,
                args.train_data_paths[lang],
                args.model_type,
                padding,
                mono_alpha,
                cross_alpha,
                args.overwrite_cache
            )
        
        if args.train_data_percentage != 1:
            data_size = len(train_dataset[list(train_dataset.keys())[0]])
            random_bool = [random.randint(0, 1) for _ in range(data_size)]
            
            for index, lang in enumerate(train_dataset):
                dataset = train_dataset[lang]
                rand_indices = [i for i in range(data_size) if random_bool[i] == index]
                if args.dataset_name == "ruletaker" or args.dataset_name == "lot":
                    dataset.context_encodings = {key: dataset.context_encodings[key][rand_indices] for key in dataset.context_encodings}
                    dataset.question_encodings = {key: dataset.question_encodings[key][rand_indices] for key in dataset.question_encodings}
                elif args.dataset_name == "leap_ids" or args.dataset_name == "ruletaker_ids":
                    dataset.encodings = {key: dataset.encodings[key][rand_indices] for key in dataset.encodings}
                dataset.labels = [dataset.labels[i] for i in rand_indices]
                train_dataset[lang] = dataset
            
    if args.do_eval:
        random.seed(65)
        np.random.seed(65)
        eval_dataset = {}
        
        for lang in args.val_data_paths:
            if "-" in lang:
                mono_alpha = args.cross_mono_eval_alpha
            else:
                mono_alpha = args.mono_eval_alpha
            cross_alpha = args.cross_eval_alpha
            
            
            eval_dataset[lang] = DATASET_CLASSES[args.dataset_name](
                tokenizer,
                args.val_data_paths[lang],
                args.model_type,
                padding,
                mono_alpha,
                cross_alpha,
                args.overwrite_cache
            )
            
        if args.train_data_percentage != 1:
            data_size = len(eval_dataset[list(eval_dataset.keys())[0]])
            random_bool = [random.randint(0, 1) for _ in range(data_size)]
            
            for index, lang in enumerate(eval_dataset):
                dataset = eval_dataset[lang]
                rand_indices = [i for i in range(data_size) if random_bool[i] == index]
                if args.dataset_name == "ruletaker" or args.dataset_name == "lot":
                    dataset.context_encodings = {key: dataset.context_encodings[key][rand_indices] for key in dataset.context_encodings}
                    dataset.question_encodings = {key: dataset.question_encodings[key][rand_indices] for key in dataset.question_encodings}
                elif args.dataset_name == "leap_ids" or args.dataset_name == "ruletaker_ids":
                    dataset.encodings = {key: dataset.encodings[key][rand_indices] for key in dataset.encodings}
                dataset.labels = [dataset.labels[i] for i in rand_indices]
                eval_dataset[lang] = dataset
        
    # Get the metric function
    random.seed(args.seed)
    np.random.seed(args.seed)
    metric = evaluate_metrics.load("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if args.pad_to_max_length:
        data_collator = default_data_collator
    elif args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    if args.do_train:
        args.logging_steps = int((sum([len(train_dataset[k]) for k in train_dataset]) / (args.per_device_train_batch_size * args.n_gpu)) / 15)
    args.eval_steps = args.logging_steps

    model.to(args.device)

    # Training
    if args.do_train:
        logger.info(
            "************************************************************************************\n"
        )
        logger.info(f"Dataset: {args.dataset_name}")
        if "ruletaker" in args.dataset_name:
            logger.info(f"Rule taker depth level: {args.rule_taker_depth_level}")
        logger.info(f"Model: {args.model_type}")
        logger.info(f"Language 1: {args.language1}")
        logger.info(f"Language 2: {args.language1}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Warmup Ratio: {args.warmup_ratio}")
        logger.info(f"Output dir: {args.output_dir}")
        logger.info(f"Train batch: {args.per_device_train_batch_size}")
        logger.info(f"Logging dir (tensorboard): {args.logging_dir}")
        logger.info(
            "************************************************************************************\n"
        )

        args.save_steps = math.ceil(int(sum([len(train_dataset[k]) for k in train_dataset]) / (args.per_device_train_batch_size * args.n_gpu)))
        print(f"args.logging_steps: {args.logging_steps}")
        print(f"args.save_steps: {args.save_steps}\n")

        
        global_step, tr_loss = train(
            args,
            train_dataset,
            eval_dataset,
            model,
            data_collator,
            compute_metrics,
            tokenizer,
        )

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        results = evaluate(args, eval_dataset, model, data_collator, compute_metrics)
        print(f"Final eval result: {results}")
        print("Execution Time: {:.2f} minutes.".format((time.time() - start) / 60))
    

if __name__ == "__main__":
    main()
