#!/usr/bin/env python
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
""" Finetuning multilingual models on reasoning tasks (e.g. Bert, DistilBERT, XLM).
    Adapted from `examples/text-classification/run_glue.py`"""

import logging
import os
import sys
import json
import random
from dataclasses import dataclass, field
from typing import Optional

import wandb

import datasets
import numpy as np
import evaluate as evaluate_metrics
from data_processing import RuleTakerDataset, LeapOfThoughtDataset
from torch.optim import AdamW
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
)
from transformers.integrations import TensorBoardCallback, WandbCallback
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

DATASET_CLASSES = {"ruletaker": RuleTakerDataset, "lot":LeapOfThoughtDataset}

MODEL_PATH = {
    "mbert": "bert-base-multilingual-cased", 
    "xlm-r": "xlm-roberta-base", 
    "xlm-r-large": "xlm-roberta-large", 
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "xlm-tlm": "xlm-mlm-tlm-xnli15-1024",
    "xlm": "xlm-mlm-xnli15-1024",
}

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


class XLMRobertaPooler(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = torch.nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    dataset_name: Optional[str] = field(
        default="ruletaker", metadata={"help": "The dataset name."}
    )
    randomized_dataset: bool = field(
        default=True, metadata={"help": "Whether to use randomized dataset fro LOT."}
    )
    data_base_dir: Optional[str] = field(metadata={"help": "The base directory of the data."}
    )
    train_data_path: Optional[str] = field(
        default=None, metadata={"help": "The path to the training data."},
    )
    val_data_path: Optional[str] = field(
        default=None, metadata={"help": "The path to the validation data."},
    )
    test_data_path: Optional[str] = field(
        default=None, metadata={"help": "The path to the test data."},
    )
    rule_taker_depth_level: Optional[str] = field(
        default=None, metadata={"help": "The depth level of the ruletaker dataset."},
    )
    logging_tool: Optional[str] = field(
        default="tensorboard", metadata={"help": "The logging tool to use."},
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    from_pretrained_model: bool = field(default=False, metadata={"help": "Whether to use pretrained model or not."})
    pretained_model_path: Optional[str] = field(default=None, metadata={"help": "The path to the pretrained model."})
    bitfit: bool = field(default=False, metadata={"help": "Whether to use bitfit or not."})
    model_type: str = field(
        default="mbert", metadata={"help": "Model type selected in the list of pre-defined models."}
    )
    model_name_or_path: str = field(
        default="bert-base-multilingual-cased", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    language: str = field(
        default=None, metadata={"help": "Evaluation language. Also train language if `train_language` is set to None."}
    )
    train_language: Optional[str] = field(
        default=None, metadata={"help": "Train language if it is different from the evaluation language."}
    )
    train_second_language: Optional[str] = field(
        default=None, metadata={"help": "Second train language if the setup is mixing two datasets."}
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
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

def bert_custom_data_collator(features):
    """ A data collator for mBERT model. """
    batch = default_data_collator(features)
    max_seq_length = batch["attention_mask"].sum(dim=1).max().item()

    batch["input_ids"] = batch["input_ids"][:, :max_seq_length]
    batch["attention_mask"] = batch["attention_mask"][:, :max_seq_length]
    batch["token_type_ids"] = batch["token_type_ids"][:, :max_seq_length]
    
    return batch

def xlm_custom_data_collator(features):
    """ A data collator for XLM-Roberta model. """
    batch = default_data_collator(features)
    max_seq_length = batch["attention_mask"].sum(dim=1).max().item()

    batch["input_ids"] = batch["input_ids"][:, :max_seq_length]
    batch["attention_mask"] = batch["attention_mask"][:, :max_seq_length]
    
    return batch

def mix_two_datasets(tokenizer, padding, train_languages, data_args, model_args, randomized=True):
    """ Load and mix two datasets (half from each dataset).
    Args:
        tokenizer: tokenizer to use
        padding: padding strategy
        train_languages: list of train languages (here only two languages are supported)
        data_args: data arguments
        model_args: model arguments
        randomized: whether to use randomized or original dataset
    Returns:
        train_dataset: mixed train dataset
        eval_dataset: mixed eval dataset
    """
    
    # Preparing Train Dataset
    random.seed(65)
    t_datasets = {}
    for lang in train_languages:
        if "ruletaker" in data_args.dataset_name:
            d_path = os.path.join(data_args.data_base_dir, lang, "original", f"depth-{data_args.rule_taker_depth_level}", "train.jsonl")
        elif "lot" in data_args.dataset_name:
            if randomized:
                d_path = os.path.join(data_args.data_base_dir, model_args.train_language, "randomized_hypernyms_training_mix_short_train.jsonl")
            else:
                d_path = os.path.join(data_args.data_base_dir, model_args.train_language, "hypernyms_training_mix_short_train.jsonl")
        t_datasets[lang] = DATASET_CLASSES[data_args.dataset_name](
        tokenizer, d_path, model_args.model_type, padding, model_args.train_language, data_args.overwrite_cache)
        
    data_size = len(t_datasets[list(t_datasets.keys())[0]])
    random_bool = [random.randint(0, 1) for _ in range(data_size)]
        
    all_t_datasets = []
    for index, lang in enumerate(t_datasets):
        dataset = t_datasets[lang]
        rand_indices = [i for i in range(data_size) if random_bool[i] == index]
        dataset.encodings = {key: dataset.encodings[key][rand_indices] for key in dataset.encodings}
        dataset.labels = [dataset.labels[i] for i in rand_indices]
        all_t_datasets.append(dataset)
    
    train_dataset = all_t_datasets[0]
    train_dataset.encodings = {key: torch.cat([train_dataset.encodings[key],  all_t_datasets[1].encodings[key]], dim=0) for key in train_dataset.encodings}
    train_dataset.labels = train_dataset.labels + all_t_datasets[1].labels
    
    # Prepare Eval Dataset
    random.seed(65)
    e_datasets = {}
    for lang in train_languages:
        if "ruletaker" in data_args.dataset_name:
            d_path = os.path.join(data_args.data_base_dir, lang, "original", f"depth-{data_args.rule_taker_depth_level}", "dev.jsonl")
        elif "lot" in data_args.dataset_name:
            if randomized:
                d_path = os.path.join(data_args.data_base_dir, model_args.language,  "randomized_hypernyms_training_mix_short_dev.jsonl")
            else:
                d_path = os.path.join(data_args.data_base_dir, model_args.language,  "hypernyms_training_mix_short_dev.jsonl")
        e_datasets[lang] = DATASET_CLASSES[data_args.dataset_name](
        tokenizer, d_path, model_args.model_type, padding, model_args.train_language, data_args.overwrite_cache)
        
    data_size = len(e_datasets[list(e_datasets.keys())[0]])
    random_bool = [random.randint(0, 1) for _ in range(data_size)]
        
    all_e_datasets = []
    for index, lang in enumerate(e_datasets):
        dataset = e_datasets[lang]
        rand_indices = [i for i in range(data_size) if random_bool[i] == index]
        
        dataset.encodings = {key: dataset.encodings[key][rand_indices] for key in dataset.encodings}
        dataset.labels = [dataset.labels[i] for i in rand_indices]
        all_e_datasets.append(dataset)
    
    eval_dataset = all_e_datasets[0]
    eval_dataset.encodings = {key: torch.cat([eval_dataset.encodings[key],  all_e_datasets[1].encodings[key]], dim=0) for key in eval_dataset.encodings}
    eval_dataset.labels = eval_dataset.labels + all_e_datasets[1].labels
    
    return train_dataset, eval_dataset
    

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()[0]
    
    if data_args.logging_tool == "wandb":
        wandb.init(project=f"{data_args.dataset}-{model_args.model_type} finetuning.")
        training_args.report_to = "wandb"
    else:
        training_args.report_to.remove("wandb")
        os.environ["WANDB_DISABLED"] = "true"

        
    if model_args.train_second_language is not None:
        train_languages = [model_args.train_language, model_args.train_second_language]
        
    model_args.model_name_or_path = MODEL_PATH[model_args.model_type]
    model_args.config_name = model_args.model_name_or_path
    model_args.tokenizer_name = model_args.model_name_or_path

    if model_args.language is None:
        model_args.language = model_args.train_language
    
    if data_args.rule_taker_depth_level == "5" and training_args.do_train:
        ValueError("5 is not a valid depth level for the training!")

    
    if "ruletaker" in data_args.dataset_name:
        data_args.test_data_path = os.path.join(data_args.data_base_dir, model_args.language, "original", f"depth-{data_args.rule_taker_depth_level}", "test.jsonl")
        data_args.val_data_path = os.path.join(data_args.data_base_dir, model_args.language, "original", f"depth-{data_args.rule_taker_depth_level}", "dev.jsonl")
        data_args.train_data_path = os.path.join(data_args.data_base_dir, model_args.train_language, "original", f"depth-{data_args.rule_taker_depth_level}", "train.jsonl")
        training_args.output_dir += f"-{data_args.rule_taker_depth_level}"
        
    elif "lot" in data_args.dataset_name:
        if data_args.randomized_dataset:
            data_args.train_data_path = os.path.join(data_args.data_base_dir, model_args.train_language, "randomized_hypernyms_training_mix_short_train.jsonl")
            data_args.val_data_path = os.path.join(data_args.data_base_dir, model_args.language,  "randomized_hypernyms_training_mix_short_dev.jsonl")
        else:
            data_args.train_data_path = os.path.join(data_args.data_base_dir, model_args.train_language, "hypernyms_training_mix_short_train.jsonl")
            data_args.val_data_path = os.path.join(data_args.data_base_dir, model_args.language,  "hypernyms_training_mix_short_dev.jsonl")
    else:
        ValueError("Please specify a valid dataset name.")
        
    label_list = [1, 0]

    # Labels
    num_labels = len(label_list)
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry(f"run_reasoning_{model_args.model_type}", model_args)
    
        
    if data_args.logging_tool == "tensorboard":
        if training_args.do_train:
            tb_writer = SummaryWriter(log_dir=training_args.logging_dir)
        else:
            tb_writer = None
            training_args.report_to.remove("tensorboard")

    logger.info("************************************************************************************\n")
    logger.info(f"Dataset: {data_args.dataset_name}")
    if data_args.dataset_name == "ruletaker":
        logger.info(f"Rule taker depth level: {data_args.rule_taker_depth_level}")
    logger.info(f"Model: {model_args.model_type}")
    logger.info(f"Language: {model_args.train_language}")
    logger.info(f"Learning Rate: {training_args.learning_rate}")
    logger.info(f"Warmup Steps: {training_args.warmup_steps}")
    logger.info(f"Warmup Ratio: {training_args.warmup_ratio}")
    logger.info(f"Logging dir: {training_args.logging_dir}")
    logger.info(f"Output dir: {training_args.output_dir}")
    logger.info(f"Seed: {training_args.seed}")
    logger.info("************************************************************************************\n")
        
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    if "xlm-r" in model_args.model_name_or_path:
        xlm_lm_model = transformers.AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path)
        pretrained_pooler_module = xlm_lm_model.lm_head.dense
        model.classifier.dense = pretrained_pooler_module
    
    if model_args.from_pretrained_model:
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        logger.info(f"\nLoading model from {model_args.pretained_model_path}\n")
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        base_model = torch.load(model_args.pretained_model_path)
        if model_args.model_type == "xlm-r":
            model.roberta.encoder.load_state_dict(base_model.roberta.encoder.state_dict(), strict=False)
            model.roberta.embeddings.load_state_dict(base_model.roberta.embeddings.state_dict(), strict=False)
        else:
            model.bert.encoder.load_state_dict(base_model.bert.encoder.state_dict(), strict=False)
            model.bert.embeddings.load_state_dict(base_model.bert.embeddings.state_dict(), strict=False)

    if model_args.bitfit:
        for name, param in model.named_parameters():
            if ".bias" in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        if model_args.model_type == "xlm-r" or "roberta" in model_args.model_type:            
            model.classifier.dense.weight.requires_grad = True
            model.classifier.dense.bias.requires_grad = True
            model.classifier.out_proj.weight.requires_grad = True
            model.classifier.out_proj.bias.requires_grad = True
            
        if model_args.model_type == "bert":
            model.bert.pooler.dense.weight.requires_grad = True
            model.bert.pooler.dense.bias.requires_grad = True
            model.classifier.weight.requires_grad = True
            model.classifier.bias.requires_grad = True

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    
    # Load datasets
    if model_args.train_second_language  is not None:
        logger.info(f"Loading mix-fixed dataset... {train_languages}")
        train_dataset, eval_dataset = mix_two_datasets(tokenizer, padding, train_languages, data_args, model_args, data_args.randomized_dataset)

    else:
        if training_args.do_train:
            train_dataset = DATASET_CLASSES[data_args.dataset_name](
                tokenizer, data_args.train_data_path, model_args.model_type, padding, model_args.train_language, data_args.overwrite_cache)

        if training_args.do_eval:
            eval_dataset = DATASET_CLASSES[data_args.dataset_name](
                tokenizer, data_args.val_data_path, model_args.model_type, padding, model_args.language, data_args.overwrite_cache)

        if training_args.do_predict:
            predict_dataset = DATASET_CLASSES[data_args.dataset_name](
                tokenizer, data_args.test_data_path, model_args.model_type, padding, model_args.language, data_args.overwrite_cache)

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # Get the metric function
    metric = evaluate_metrics.load("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        # data_collator = default_data_collator
        if model_args.model_type == "mbert":
            data_collator = bert_custom_data_collator
        elif "xlm" in model_args.model_type or "roberta" in model_args.model_type:
            data_collator = xlm_custom_data_collator
        else:
            ValueError("Please specify a data collator for your model type")
            
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    if training_args.do_train:
        training_args.logging_steps = int((len(train_dataset) / training_args.per_device_train_batch_size * training_args.n_gpu)/15)
    training_args.eval_steps = training_args.logging_steps
    
        
    t_total = ((len(train_dataset) // training_args.per_device_train_batch_size) * training_args.num_train_epochs)

        # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate, eps=1e-8, weight_decay=training_args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=t_total)
        
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler) 
    )
    
    if data_args.logging_tool == "tensorboard" and tb_writer is not None:
        trainer.add_callback(TensorBoardCallback(tb_writer=tb_writer))
    elif data_args.logging_tool == "wandb":
        trainer.add_callback(WandbCallback())

    # Training
    if training_args.do_train:
        # Save experiment args
        with open(os.path.join(training_args.output_dir, "training_args.json"), 'w') as fp:
            json.dump({k:str(v) for k,v in vars(training_args).items()}, fp)
        with open(os.path.join(training_args.output_dir, "data_args.json"), 'w') as fp:
            json.dump({k:str(v) for k,v in vars(data_args).items()}, fp)
        with open(os.path.join(training_args.output_dir, "model_args.json"), 'w') as fp:
            json.dump({k:str(v) for k,v in vars(model_args).items()}, fp)
        
        logger.info("************************************************************************************\n")
        logger.info(f"Dataset: {data_args.dataset_name}")
        if data_args.dataset_name == "ruletaker":
            logger.info(f"RuleTaker depth level: {data_args.rule_taker_depth_level}")
        logger.info(f"Model: {model_args.model_type}")
        if model_args.train_second_language is not None:
            logger.info(f"Train languages: {train_languages}")
        else:
            logger.info(f"Train Language: {model_args.train_language}")
        logger.info(f"Learning rate: {training_args.learning_rate}")
        logger.info(f"Warmup Steps: {training_args.warmup_steps}")
        logger.info(f"Warmup Ratio: {training_args.warmup_ratio}")
        logger.info(f"Output dir: {training_args.output_dir}")
        logger.info(f"Logging dir (tensorboard): {training_args.logging_dir}")
        logger.info(f"training_args.report_to: {training_args.report_to}")
        logger.info("************************************************************************************\n")
                
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        if data_args.logging_tool == "wandb":
            wandb.finish()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        print(f"Eval metrics: {metrics}")


if __name__ == "__main__":
    main()