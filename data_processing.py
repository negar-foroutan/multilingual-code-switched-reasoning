import os
import json
import time
import logging
import pickle
import random
import numpy as np

import nltk
from googletrans import Translator
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RuleTakerDataset(torch.utils.data.Dataset):
    """ A Dataset class to load RuleTaker data. """
    def __init__(self, tokenizer, data_file_path, model_type, padding, overwrite_cache=False):
        """ Initializes the dataset object .

        Args:
            tokenizer (transformers.AutoTokenizer): The tokenizer to tokenize the text.
            data_file_path (str): Data file path.
            model_type (str): Language model type.
            padding (bool/str): Whether to pad the sequence or not.
            overwrite_cache (bool, optional): Wether overwrite the data cache file. Defaults to False.
        """
        
        if "bert" in model_type:
            sep_token = "[SEP]"
        if "xlm" in model_type or "roberta" in model_type:
            sep_token = "</s>"
    

        directory, filename = os.path.split(data_file_path)
        cached_features_file = os.path.join(
            directory, model_type + "_cached_" + str(tokenizer.model_max_length) + "_" + filename.split(".")[0] + ".pkl"
        )

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                examples = pickle.load(handle)
                self.encodings = examples[0]
                self.labels = examples[1]
        else:
            logger.info(f"Loading data from {data_file_path}")
            start = time.time()
            self.encodings = []
            self.labels = []
            texts = []
            with open(data_file_path, 'r') as reader:
                for index, line in enumerate(reader):
                    sample = json.loads(line)
                    context = sample['context']
                    questions = sample['questions']
                    for q in questions:
                        statement = q["text"]
                        input_text = context + f" {sep_token} " + statement 
                        texts.append(input_text)
                        self.labels.append(1 if q["label"] else 0)
                    if index % 2000 == 0:
                        logger.info(f"Processed {index} lines")
                        
            self.encodings = tokenizer(texts,  padding=padding, truncation=True, return_tensors='pt')
            logger.info(f"Number of samples: {len(self.labels)}, Time: {time.time() - start} seconds.")

            logger.info(f"Saving features into cached file {cached_features_file}")
            examples = [self.encodings, self.labels]
            with open(cached_features_file, "wb") as handle:
                pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
class LeapOfThoughtDataset(torch.utils.data.Dataset):
    """ A Dataset class to load LeapOfThought data. """
    def __init__(self, tokenizer, data_file_path, model_type, padding, overwrite_cache=False):
        """ Initializes the dataset object .

        Args:
            tokenizer (transformers.AutoTokenizer): The tokenizer to tokenize the text.
            data_file_path (str): Data file path.
            model_type (str): Language model type.
            padding (bool/str): Whether to pad the sequence or not.
            overwrite_cache (bool, optional): Wether overwrite the data cache file. Defaults to False.
        """
        
        if "bert" in model_type:
            sep_token = "[SEP]"
        if "xlm" in model_type or "roberta" in model_type:
            sep_token = "</s>"

        directory, filename = os.path.split(data_file_path)
        cached_features_file = os.path.join(
            directory, model_type + "_cached_" + str(tokenizer.model_max_length) + "_" + filename.split(".")[0] + ".pkl"
        )

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                examples = pickle.load(handle)
                self.encodings = examples[0]
                self.labels = examples[1]
        else:
            logger.info(f"Loading data from {data_file_path}")
            start = time.time()
            self.encodings = []
            self.labels = []
            texts = []
            with open(data_file_path, 'r') as reader:
                for index, line in enumerate(reader):
                    sample = json.loads(line)
                    context = sample['context']
                    statement = sample['phrase']
                    input_text = context + f" {sep_token} " + statement
                    input_text = input_text.strip()
                    texts.append(input_text)
                    self.labels.append(int(sample["answer"]))
                    if index % 2000 == 0:
                        logger.info(f"Processed {index} lines")
                        
            self.encodings = tokenizer(texts,  padding=padding, return_tensors='pt')
            examples = [self.encodings, self.labels]
            
            logger.info(f"Number of samples: {len(self.labels)}, Time: {time.time() - start} seconds.")
            logger.info(f"Saving features into cached file {cached_features_file}")
            with open(cached_features_file, "wb") as handle:
                pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class RuleTakerCurriculumDataset(torch.utils.data.Dataset):
    """ A Dataset class to load RuleTaker data for Curriculum learning. """

    def __init__(self, tokenizer, data_file_path, model_type, padding, overwrite_cache=False):
        """ Initializes the dataset object .

        Args:
            tokenizer (transformers.AutoTokenizer): The tokenizer to tokenize the text.
            data_file_path (str): Data file path.
            model_type (str): Language model type.
            padding (bool/str): Whether to pad the sequence or not.
            overwrite_cache (bool, optional): Wether overwrite the data cache file. Defaults to False.
        """

        if "bert" in model_type:
            sep_token = "[SEP]"
        if "xlm" in model_type:
            sep_token = "</s>"

        directory, filename = os.path.split(data_file_path)
        cached_features_file = os.path.join(
            directory, model_type + "_cached_" + str(tokenizer.model_max_length) + "_curriculum_" + filename.split(".")[0] + ".pkl"
        )

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                examples = pickle.load(handle)
                self.encodings = examples[0]
                self.labels = examples[1]
                self.depths = examples[2]

        logger.info(f"Loading data from {data_file_path}")
        start = time.time()
        self.encodings = []
        self.labels = []
        self.depths = []
        texts = []
        with open(data_file_path, 'r') as reader:
            for index, line in enumerate(reader):
                sample = json.loads(line)
                context = sample['context']
                questions = sample['questions']
                for q in questions:
                    statement = q["text"]
                    input_text = context + f" {sep_token} " + statement
                    texts.append(input_text)
                    self.labels.append(1 if q["label"] else 0)
                    self.depths.append(q["meta"]["QDep"])
                if index % 2000 == 0:
                    logger.info(f"Processed {index} lines")

        self.encodings = tokenizer(texts, padding=padding, return_tensors='pt')
        logger.info(f"Number of samples: {len(self.labels)}, Time: {time.time() - start} seconds.")

        logger.info(f"Saving features into cached file {cached_features_file}")
        examples = [self.encodings, self.labels, self.depths]
        with open(cached_features_file, "wb") as handle:
            pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class RuleTakerDatasetSepAttentionMask(torch.utils.data.Dataset):
    """ A Dataset class to load RuleTaker data with having separate attention masks for monolingual and cross-lingual attentions. """
    def __init__(self, tokenizer, data_file_path, model_type, padding, mono_alpha,
                 cross_alpha, overwrite_cache=False):
        """ Initializes the dataset object.

        Args:
            tokenizer (transformers.AutoTokenizer): The tokenizer to tokenize the text.
            data_file_path (str): Data file path.
            model_type (str): Language model type.
            padding (bool/str): Whether to pad the sequence or not.
            mono_alpha: Probability of attention between two tokens from the same language using the monolingual Query.
            cross_alpha: Probability of attention between two tokens from different languages using the cross-lingual Query.
            overwrite_cache (bool, optional): Wether overwrite the data cache file. Defaults to False.
        """

        directory, filename = os.path.split(data_file_path)
        cached_features_file = os.path.join(
            directory, model_type + "_cached_with_sep_attn_mask_" + str(tokenizer.model_max_length) + "_" + filename.split(".")[0] + ".pkl"
        )

        self.max_seq_length = tokenizer.model_max_length
        self.mono_alpha = mono_alpha
        self.cross_alpha = cross_alpha
        
        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                examples = pickle.load(handle)
                self.context_encodings = examples[0]
                self.question_encodings = examples[1]
                self.labels = examples[2]
        else:
            logger.info("Creating features from dataset file at %s", data_file_path)
            start = time.time()
            self.context_encodings = []
            self.question_encodings = []
            self.labels = []
            contexts = []
            questions = []
            with open(data_file_path, 'r') as reader:
                for index, line in enumerate(reader):
                    sample = json.loads(line)
                    for q in sample['questions']:
                        contexts.append(sample['context'])
                        questions.append(q["text"])
                        self.labels.append(1 if q["label"] else 0)
                    if index % 2000 == 0:
                        logger.info(f"Processed {index} lines")
                        
            self.context_encodings = tokenizer(contexts,  padding=padding, return_tensors='pt')
            self.question_encodings = tokenizer(questions,  padding=padding, return_tensors='pt')
                
            examples = [self.context_encodings, self.question_encodings, self.labels]
            logger.info(f"Number of samples: {len(self.labels)}, Time: {time.time() - start} seconds.")

            logger.info(f"Saving features into cached file {cached_features_file}")
            with open(cached_features_file, "wb") as handle:
                pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, idx):
        input_ids_question = self.question_encodings['input_ids'][idx]
        attention_mask_question = self.question_encodings['attention_mask'][idx]
        
        input_ids_context = self.context_encodings['input_ids'][idx]
        attention_mask_context = self.context_encodings['attention_mask'][idx]
        
        question_lengths = attention_mask_question.sum(dim=0) - 1
        contexts_lengths = attention_mask_context.sum(dim=0)

        input_ids = torch.cat([input_ids_context[:contexts_lengths], input_ids_question[1:question_lengths+1], 
                               torch.tensor([0] * (self.max_seq_length - (contexts_lengths + question_lengths)))])
        
        mono_attention_mask = self.get_mono_attention_mask(contexts_lengths, question_lengths, self.max_seq_length, self.mono_alpha)
        cross_attention_mask = self.get_cross_lingual_attention_mask(contexts_lengths, question_lengths, self.max_seq_length, self.cross_alpha)
        
        attention_mask = torch.cat([mono_attention_mask, cross_attention_mask], dim=0)
        token_type_ids = torch.zeros(input_ids.shape[0]).long()
        item = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": torch.tensor(self.labels[idx]), "token_type_ids": token_type_ids}
        return item
        
    def __len__(self):
        return len(self.labels)

    def get_mono_attention_mask(self, contexts_length, question_length, max_seq_length, mono_alpha=0):
        """ Computes the monolingual attention mask for the given input. 
         Args:
            contexts_length: Length of the context.
            question_length: Length of the question.
            max_seq_length: Maximum sequence length.
            mono_alpha: Probability of attention between two tokens from the same language in this mask.
        """
        temp_attention_mask = torch.zeros(max_seq_length, max_seq_length)
        
        temp_attention_mask[:, :contexts_length + question_length] = torch.bernoulli(
            mono_alpha * torch.ones(max_seq_length, contexts_length + question_length))
        
        # Context to context attention
        temp_attention_mask[:contexts_length, :contexts_length] = 1
        # Question to question attention
        temp_attention_mask[contexts_length : contexts_length + question_length,
                            contexts_length : contexts_length + question_length] = 1
        
        # CLS to every token
        temp_attention_mask[0, :contexts_length + question_length] = 1
        # Every token to CLS
        temp_attention_mask[:, 0] = 1

        return temp_attention_mask.unsqueeze(0)
        
    def get_cross_lingual_attention_mask(self, contexts_length, question_length, max_seq_length, cross_alpha=0):
        """ Computes the cross-lingual attention mask for the given input. 
         Args:
            contexts_length: Length of the context.
            question_length: Length of the question.
            max_seq_length: Maximum sequence length.
            cross_alpha: Probability of attention between two tokens from different languages in this mask.
        """
        temp_attention_mask = torch.zeros(max_seq_length, max_seq_length)
        
        temp_attention_mask[:, :contexts_length + question_length] = torch.bernoulli(
            cross_alpha * torch.ones(max_seq_length, contexts_length + question_length))
        
        # Context to question attention 
        temp_attention_mask[:contexts_length, contexts_length : contexts_length + question_length] = 1
        # Question to context attention
        temp_attention_mask[contexts_length : contexts_length + question_length, :contexts_length] = 1
        
        # CLS to every token
        temp_attention_mask[0, :contexts_length + question_length] = 1
        # Every token to CLS
        temp_attention_mask[:, 0] = 1

        return temp_attention_mask.unsqueeze(0)
        

class LeapOfThoughtDatasetSepAttentionMask(torch.utils.data.Dataset):
    """ A Dataset class to load LeapOfThought data with having separate attention masks for monolingual and cross-lingual attentions. """
    def __init__(self, tokenizer, data_file_path, model_type, padding, mono_alpha, cross_alpha, overwrite_cache=False):
        """ Initializes the dataset object.

        Args:
            tokenizer (transformers.AutoTokenizer): The tokenizer to tokenize the text.
            data_file_path (str): Data file path.
            model_type (str): Language model type.
            padding (bool/str): Whether to pad the sequence or not.
            mono_alpha: Probability of attention between two tokens from the same language using the monolingual Query.
            cross_alpha: Probability of attention between two tokens from different languages using the cross-lingual Query.
            overwrite_cache (bool, optional): Wether overwrite the data cache file. Defaults to False.
        """

        directory, filename = os.path.split(data_file_path)
        cached_features_file = os.path.join(
            directory, model_type + "_cached_with_sep_attn_mask_" + str(tokenizer.model_max_length) + "_" + filename.split(".")[0] + ".pkl"
        )
        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                examples = pickle.load(handle)
                self.context_encodings = examples[0]
                self.question_encodings = examples[1]
                self.labels = examples[2]

        self.max_seq_length = tokenizer.model_max_length
        self.mono_alpha = mono_alpha
        self.cross_alpha = cross_alpha
        self.sep_token_id = tokenizer.sep_token_id
        
        start = time.time()
        self.context_encodings = []
        self.question_encodings = []
        self.labels = []
        contexts = []
        questions = []
        with open(data_file_path, 'r') as reader:
            for index, line in enumerate(reader):
                sample = json.loads(line)
                contexts.append(sample['context'])
                questions.append(sample["phrase"])
                self.labels.append(sample["answer"])
                if index % 2000 == 0:
                    logger.info(f"Processed {index} lines")
        
        self.context_encodings = tokenizer(contexts,  padding=padding, return_tensors='pt')
        self.question_encodings = tokenizer(questions,  padding=padding, return_tensors='pt')
        logger.info(f"Number of samples: {len(self.labels)}, Time: {time.time() - start} seconds.")

        logger.info(f"Saving features into cached file {cached_features_file}")
        examples = [self.context_encodings,  self.question_encodings, self.labels]
        with open(cached_features_file, "wb") as handle:
            pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    def __getitem__(self, idx):
        input_ids_question = self.question_encodings['input_ids'][idx]
        attention_mask_question = self.question_encodings['attention_mask'][idx]
        
        input_ids_context = self.context_encodings['input_ids'][idx]
        attention_mask_context = self.context_encodings['attention_mask'][idx]
        
        question_lengths = attention_mask_question.sum(dim=0) - 1
        contexts_lengths = attention_mask_context.sum(dim=0)

        input_ids = torch.cat([input_ids_context[:contexts_lengths], input_ids_question[1:question_lengths+1], 
                               torch.tensor([0] * (self.max_seq_length - (contexts_lengths + question_lengths)))])
        
        mono_attention_mask = self.get_mono_attention_mask(contexts_lengths, question_lengths, self.max_seq_length, self.mono_alpha)
        cross_attention_mask = self.get_cross_lingual_attention_mask(contexts_lengths, question_lengths, self.max_seq_length, self.cross_alpha)
        
        attention_mask = torch.cat([mono_attention_mask, cross_attention_mask], dim=0)
        token_type_ids = torch.zeros(input_ids.shape[0]).long()
        item = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": torch.tensor(self.labels[idx]), "token_type_ids": token_type_ids}
        return item
        
    def __len__(self):
        return len(self.labels)

    def get_mono_attention_mask(self, contexts_length, question_length, max_seq_length, mono_alpha=0):
        """ Computes the monolingual attention mask for the given input. 
         Args:
            contexts_length: Length of the context.
            question_length: Length of the question.
            max_seq_length: Maximum sequence length.
            mono_alpha: Probability of attention between two tokens from the same language in this mask.
        """
        
        temp_attention_mask = torch.zeros(max_seq_length, max_seq_length)
        
        temp_attention_mask[:, :contexts_length + question_length] = torch.bernoulli(
            mono_alpha * torch.ones(max_seq_length, contexts_length + question_length))
        
        # Context to context attention
        temp_attention_mask[:contexts_length, :contexts_length] = 1
        # Question to question attention
        temp_attention_mask[contexts_length : contexts_length + question_length,
                            contexts_length : contexts_length + question_length] = 1
        
        # CLS to every token
        temp_attention_mask[0, :contexts_length + question_length] = 1
        # Every token to CLS
        temp_attention_mask[:, 0] = 1

        return temp_attention_mask.unsqueeze(0)

    def get_cross_lingual_attention_mask(self, contexts_length, question_length, max_seq_length, cross_alpha=0):
        """ Computes the cross-lingual attention mask for the given input. 
         Args:
            contexts_length: Length of the context.
            question_length: Length of the question.
            max_seq_length: Maximum sequence length.
            cross_alpha: Probability of attention between two tokens from different languages in this mask.
        """
        temp_attention_mask = torch.zeros(max_seq_length, max_seq_length)
        
        temp_attention_mask[:, :contexts_length + question_length] = torch.bernoulli(
            cross_alpha * torch.ones(max_seq_length, contexts_length + question_length))
        
        # Context to question attention 
        temp_attention_mask[:contexts_length, contexts_length : contexts_length + question_length] = 1
        # Question to context attention
        temp_attention_mask[contexts_length : contexts_length + question_length, :contexts_length] = 1
        
        # CLS to every token
        temp_attention_mask[0, :contexts_length + question_length] = 1
        # Every token to CLS
        temp_attention_mask[:, 0] = 1

        return temp_attention_mask.unsqueeze(0)


class RuleTakerCurriculumDatasetSepAttentionMask(torch.utils.data.Dataset):
    """ A Dataset class to load RuleTaker data for curriculum learning with having separate attention masks for monolingual and cross-lingual attentions. """

    def __init__(self, tokenizer, data_file_path, model_type, padding, mono_alpha, cross_alpha, overwrite_cache=False):
        """ Initializes the dataset object .

        Args:
            tokenizer (transformers.AutoTokenizer): The tokenizer to tokenize the text.
            data_file_path (str): Data file path.
            model_type (str): Language model type.
            padding (bool/str): Whether to pad the sequence or not.
            mono_alpha: Probability of attention between two tokens from the same language using the monolingual Query.
            cross_alpha: Probability of attention between two tokens from different languages using the cross-lingual Query.
            overwrite_cache (bool, optional): Whether overwrite the data cache file. Defaults to False.
        """

        directory, filename = os.path.split(data_file_path)
        cached_features_file = os.path.join(
            directory,
            model_type + "_cached_with_sep_attn_mask_" + str(tokenizer.model_max_length) + "_curriculum_" + filename.split(".")[0] + ".pkl"
        )
        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                examples = pickle.load(handle)
                self.context_encodings = examples[0]
                self.question_encodings = examples[1]
                self.labels = examples[2]
                self.depths = examples[3]

        self.max_seq_length = tokenizer.model_max_length
        self.mono_alpha = mono_alpha
        self.cross_alpha = cross_alpha

        logger.info("Creating features from dataset file at %s", data_file_path)
        start = time.time()
        self.context_encodings = []
        self.question_encodings = []
        self.labels = []
        self.depths = []
        contexts = []
        questions = []
        with open(data_file_path, 'r') as reader:
            for index, line in enumerate(reader):
                sample = json.loads(line)
                for q in sample['questions']:
                    contexts.append(sample['context'])
                    questions.append(q["text"])
                    self.labels.append(1 if q["label"] else 0)
                    self.depths.append(q["meta"]["QDep"])
                if index % 2000 == 0:
                    logger.info(f"Processed {index} lines")

        self.context_encodings = tokenizer(contexts, padding=padding, return_tensors='pt')
        self.question_encodings = tokenizer(questions, padding=padding, return_tensors='pt')
        logger.info(f"Number of samples: {len(self.labels)}, Time: {time.time() - start} seconds.")

        logger.info(f"Saving features into cached file {cached_features_file}")
        examples = [self.context_encodings, self.question_encodings, self.labels, self.depths]
        with open(cached_features_file, "wb") as handle:
            pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

            

    def __getitem__(self, idx):
        input_ids_question = self.question_encodings['input_ids'][idx]
        attention_mask_question = self.question_encodings['attention_mask'][idx]

        input_ids_context = self.context_encodings['input_ids'][idx]
        attention_mask_context = self.context_encodings['attention_mask'][idx]

        question_lengths = attention_mask_question.sum(dim=0) - 1
        contexts_lengths = attention_mask_context.sum(dim=0)

        input_ids = torch.cat([input_ids_context[:contexts_lengths], input_ids_question[1:question_lengths + 1],
                               torch.tensor([0] * (self.max_seq_length - (contexts_lengths + question_lengths)))])

        mono_attention_mask = self.get_mono_attention_mask(contexts_lengths, question_lengths, self.max_seq_length,
                                                           self.mono_alpha)
        cross_attention_mask = self.get_cross_lingual_attention_mask(contexts_lengths, question_lengths,
                                                                     self.max_seq_length, self.cross_alpha)

        attention_mask = torch.cat([mono_attention_mask, cross_attention_mask], dim=0)
        token_type_ids = torch.zeros(input_ids.shape[0]).long()
        item = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": torch.tensor(self.labels[idx]),
                "token_type_ids": token_type_ids}
        return item

    def __len__(self):
        return len(self.labels)

    def get_mono_attention_mask(self, contexts_length, question_length, max_seq_length, mono_alpha=0):
        """ Computes the monolingual attention mask for the given input. 
         Args:
            contexts_length: Length of the context.
            question_length: Length of the question.
            max_seq_length: Maximum sequence length.
            mono_alpha: Probability of attention between two tokens from the same language in this mask.
        """
        temp_attention_mask = torch.zeros(max_seq_length, max_seq_length)

        temp_attention_mask[:, :contexts_length + question_length] = torch.bernoulli(
            mono_alpha * torch.ones(max_seq_length, contexts_length + question_length))

        # Context to context attention
        temp_attention_mask[:contexts_length, :contexts_length] = 1
        # Question to question attention
        temp_attention_mask[contexts_length: contexts_length + question_length,
                            contexts_length: contexts_length + question_length] = 1

        # CLS to every token
        temp_attention_mask[0, :contexts_length + question_length] = 1
        # Every token to CLS
        temp_attention_mask[:, 0] = 1

        return temp_attention_mask.unsqueeze(0)

    def get_cross_lingual_attention_mask(self, contexts_length, question_length, max_seq_length, cross_alpha=0):
        """ Computes the cross-lingual attention mask for the given input. 
            Args:
                contexts_length: Length of the context.
                question_length: Length of the question.
                max_seq_length: Maximum sequence length.
                cross_alpha: Probability of attention between two tokens from different languages in this mask.
        """
        
        temp_attention_mask = torch.zeros(max_seq_length, max_seq_length)

        temp_attention_mask[:, :contexts_length + question_length] = torch.bernoulli(
            cross_alpha * torch.ones(max_seq_length, contexts_length + question_length))

        # Context to question attention
        temp_attention_mask[:contexts_length, contexts_length: contexts_length + question_length] = 1
        # Question to context attention
        temp_attention_mask[contexts_length: contexts_length + question_length, :contexts_length] = 1

        # CLS to every token
        temp_attention_mask[0, :contexts_length + question_length] = 1
        # Every token to CLS
        temp_attention_mask[:, 0] = 1

        return temp_attention_mask.unsqueeze(0)


class XNLICodeSwitchedSepAttentionMask(torch.utils.data.Dataset):
    """ A Dataset class to load XNLI data with having separate attention masks for monolingual and cross-lingual attentions. """
    def __init__(self, dataset1, dataset2, tokenizer, padding, mono_alpha, cross_alpha, is_mixed=False):
        """ Initializes the dataset object .

        Args:
            dataset1: XNLI dataset in the first language.
            dataset2: XNLI dataset in the second language.
            model_type (str): Language model type.
            tokenizer (transformers.AutoTokenizer): The tokenizer to tokenize the text.
            padding (bool/str): Whether to pad the sequence or not.
            mono_alpha: Probability of attention between two tokens from the same language using the monolingual Query.
            cross_alpha: Probability of attention between two tokens from different languages using the cross-lingual Query.
            is_mixed (bool, optional): If we eant a mixture of monolingual and cross-lingual data.
        """

        self.max_seq_length = tokenizer.model_max_length
        self.mono_alpha = mono_alpha
        self.cross_alpha = cross_alpha

        start = time.time()
        print(f"tokenizer.sep_token: {tokenizer.sep_token}")
        self.context_encodings = []
        self.question_encodings = []
        contexts = []
        questions = []
        self.labels = []

        dataset_size = min(len(dataset1), len(dataset2))
        logger.info(f"Dataset size: {len(dataset1)}")

        if is_mixed:
            indices = list(range(dataset_size))
            random.shuffle(indices)
            half_point = dataset_size // 2
            # Half of the dataset is code-switched
            for i in range(half_point):
                contexts.append(dataset1[indices[i]]["premise"])
                questions.append(dataset2[indices[i]]["hypothesis"])
                self.labels.append(dataset1[indices[i]]["label"])

            # Half of the dataset is from the same language
            for index in indices[half_point:]:
                contexts.append(dataset1[index]["premise"])
                questions.append(dataset1[index]["hypothesis"])
                self.labels.append(dataset1[index]["label"])
        else:
            for index in range(dataset_size):
                contexts.append(dataset1[index]["premise"])
                questions.append(dataset2[index]["hypothesis"])
                self.labels.append(dataset1[index]["label"])
        
        self.context_encodings = tokenizer(contexts,  padding=padding, return_tensors='pt')
        self.question_encodings = tokenizer(questions,  padding=padding, return_tensors='pt')        
        logger.info(f"Number of samples: {len(self.labels)}, Time: {time.time() - start} seconds.")


    def __getitem__(self, idx):
        input_ids_question = self.question_encodings['input_ids'][idx]
        attention_mask_question = self.question_encodings['attention_mask'][idx]
        
        input_ids_context = self.context_encodings['input_ids'][idx]
        attention_mask_context = self.context_encodings['attention_mask'][idx]
        
        question_lengths = attention_mask_question.sum(dim=0) - 1
        contexts_lengths = attention_mask_context.sum(dim=0)

        input_ids = torch.cat([input_ids_context[:contexts_lengths], input_ids_question[1:question_lengths+1], 
                               torch.tensor([0] * (self.max_seq_length - (contexts_lengths + question_lengths)))])
        
        mono_attention_mask = self.get_mono_attention_mask(contexts_lengths, question_lengths, self.max_seq_length, self.mono_alpha)
        cross_attention_mask = self.get_cross_lingual_attention_mask(contexts_lengths, question_lengths, self.max_seq_length, self.cross_alpha)
        
        attention_mask = torch.cat([mono_attention_mask, cross_attention_mask], dim=0)
        token_type_ids = torch.zeros(input_ids.shape[0]).long()
        item = {"input_ids": input_ids, "attention_mask": attention_mask,
                "labels": torch.tensor(self.labels[idx]), "token_type_ids": token_type_ids}
        return item
        
    def __len__(self):
        return len(self.labels)

    
    def get_mono_attention_mask(self, contexts_length, question_length, max_seq_length, mono_alpha=0):
        """ Computes the monolingual attention mask for the given input. 
         Args:
            contexts_length: Length of the context.
            question_length: Length of the question.
            max_seq_length: Maximum sequence length.
            mono_alpha: Probability of attention between two tokens from the same language in this mask.
        """
        temp_attention_mask = torch.zeros(max_seq_length, max_seq_length)
        
        temp_attention_mask[:, :contexts_length + question_length] = torch.bernoulli(
            mono_alpha * torch.ones(max_seq_length, contexts_length + question_length))
        
        # Context to context attention
        temp_attention_mask[:contexts_length, :contexts_length] = 1
        # Question to question attention
        temp_attention_mask[contexts_length : contexts_length + question_length,
                            contexts_length : contexts_length + question_length] = 1
        
        # CLS to every token
        temp_attention_mask[0, :contexts_length + question_length] = 1
        # Every token to CLS
        temp_attention_mask[:, 0] = 1

        return temp_attention_mask.unsqueeze(0)
        
    def get_cross_lingual_attention_mask(self, contexts_length, question_length, max_seq_length, cross_alpha=0):
        """ Computes the cross-lingual attention mask for the given input. 
             Args:
                contexts_length: Length of the context.
                question_length: Length of the question.
                max_seq_length: Maximum sequence length.
                cross_alpha: Probability of attention between two tokens from different languages in this mask.
        """
        temp_attention_mask = torch.zeros(max_seq_length, max_seq_length)
        
        temp_attention_mask[:, :contexts_length + question_length] = torch.bernoulli(
            cross_alpha * torch.ones(max_seq_length, contexts_length + question_length))
        
        # Context to question attention 
        temp_attention_mask[:contexts_length, contexts_length : contexts_length + question_length] = 1
        # Question to context attention
        temp_attention_mask[contexts_length : contexts_length + question_length, :contexts_length] = 1
        
        # CLS to every token
        temp_attention_mask[0, :contexts_length + question_length] = 1
        # Every token to CLS
        temp_attention_mask[:, 0] = 1

        return temp_attention_mask.unsqueeze(0)

class XNLIPretrainingDataset(torch.utils.data.Dataset):
    """ A Dataset class to load XNLI data for pre-training cross-lingual query. """
    def __init__(self, language_pair, dataset1, dataset2, tokenizer, padding, max_seq_length, mono_alpha, cross_alpha,
                 is_parallel=True, truncation=True, code_switched_format=None):
        """ Initializes the dataset object .

        Args:
            language_pair: Language pair for the dataset.
            dataset1: XNLI dataset in the first language.
            dataset2: XNLI dataset in the second language.
            tokenizer (transformers.AutoTokenizer): The tokenizer to tokenize the text.
            padding (bool/str): Whether to pad the sequence or not.
            max_seq_length: Maximum sequence length.
            mono_alpha: Probability of attention between two tokens from the same language using the monolingual Query.
            cross_alpha: Probability of attention between two tokens from different languages using the cross-lingual Query.
            is_parallel: Whether to use the parallel setup or not.
            truncation: Whether to truncate the sequences or not.
            code_switched_format: Format of the code-switched data (en-x, x-en, or x-x).
            overwrite_cache (bool, optional): Wether overwrite the data cache file. Defaults to False.
        """
            
        start = time.time()
        lt = time.time()
        lang1, lang2 = language_pair.split("-")
        
        self.encodings1 = []
        self.encodings2 = []
        
        self.max_seq_length = max_seq_length
        self.mono_alpha = mono_alpha
        self.cross_alpha = cross_alpha

        text1 = []
        text2 = []

        dataset_size = min(len(dataset1), len(dataset2))
        logger.info(f"Preparing XNLI dataset fot pre-training WITHOUT token type ids.")
        logger.info(f"Dataset size: {len(dataset1)}")
        feature_1 = "premise"
        if is_parallel:
            feature_2 = "premise"
        else:
            feature_2 = "hypothesis"

        for index in range(dataset_size):
            t1 = dataset1[index][feature_1]
            t2 = dataset2[index][feature_2]
            text1.append(t1)
            text2.append(t2)
            if index % 50000 == 0 and index > 0:
                logger.info(f"Processed {index}({(index * 100.0) / dataset_size:.2f}%) examples in {(time.time() - lt):.2f} seconds.")
                lt = time.time()
                self.encodings1.append(tokenizer(text1,  padding=padding, max_length=max_seq_length,
                                                 return_tensors='pt', truncation=truncation))
                self.encodings2.append(tokenizer(text2,  padding=padding, max_length=max_seq_length,
                                                 return_tensors='pt', truncation=truncation))
                text1 = []
                text2 = []

        if len(text1) > 0:
            self.encodings1.append(tokenizer(text1,  padding=padding, max_length=max_seq_length,
                                             return_tensors='pt', truncation=truncation))
            self.encodings2.append(tokenizer(text2,  padding=padding, max_length=max_seq_length,
                                             return_tensors='pt', truncation=truncation))
            
        self.encodings1 = {key: torch.cat([enc[key] for enc in self.encodings1], dim=0)
                           for key, _ in self.encodings1[0].items()}
        self.encodings2 = {key: torch.cat([enc[key] for enc in self.encodings2], dim=0)
                           for key, _ in self.encodings2[0].items()}
        
        if code_switched_format == "en-x":
            if lang2 == "en":
                self.encodings1, self.encodings2 = self.encodings2, self.encodings1
                
        elif code_switched_format == "x-en":
            if lang1 == "en":
                self.encodings1, self.encodings2 = self.encodings2, self.encodings1
        elif code_switched_format == "x-x":
            temp1 = {key: [] for key in self.encodings1}
            temp2 = {key: [] for key in self.encodings1}
            for i in range(len(self.encodings1["input_ids"])):
                rand_index = random.randint(0, 1)
                if rand_index == 0:
                    for key in self.encodings1:
                        temp1[key].append(self.encodings1[key][i])
                        temp2[key].append(self.encodings2[key][i])
                else:
                    for key in self.encodings1:
                        temp2[key].append(self.encodings1[key][i])
                        temp1[key].append(self.encodings2[key][i])
            
            self.encodings1 = temp1
            self.encodings2 = temp2
                        
        elif code_switched_format is None:
            pass
        else:
            ValueError("Invalid code_switched_format")

        dataset_size = len(self.encodings1["input_ids"])
        logger.info(f"Number of samples: {dataset_size}, Time: {time.time() - start} seconds.")

    def __getitem__(self, idx):
        input_ids_context = self.encodings1['input_ids'][idx]
        attention_mask_context = self.encodings1['attention_mask'][idx]
        
        input_ids_question = self.encodings2['input_ids'][idx]
        attention_mask_question = self.encodings2['attention_mask'][idx]
        
        question_lengths = attention_mask_question.sum(dim=0) - 1
        contexts_lengths = attention_mask_context.sum(dim=0)

        input_ids = torch.cat([input_ids_context[:contexts_lengths], input_ids_question[1:question_lengths+1], 
                            torch.tensor([0] * (self.max_seq_length - (contexts_lengths + question_lengths)))])
        
        mono_attention_mask = get_mono_attention_mask(contexts_lengths, question_lengths, self.max_seq_length, self.mono_alpha)
        cross_attention_mask = get_cross_lingual_attention_mask(contexts_lengths, question_lengths, self.max_seq_length, self.cross_alpha)
        
        attention_mask = torch.cat([mono_attention_mask, cross_attention_mask], dim=0)
        
        if input_ids.shape[0] > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            
        token_type_ids = torch.zeros(input_ids.shape[0]).long()
        item = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
        return item

    def __len__(self):
        return len(self.encodings1["input_ids"])



def get_mono_attention_mask(contexts_length, question_length, max_seq_length, mono_alpha=0):

    max_length = max(max_seq_length, contexts_length + question_length)
    temp_attention_mask = torch.zeros(max_length, max_length)
    
    temp_attention_mask[:, :contexts_length + question_length] = torch.bernoulli(
        mono_alpha * torch.ones(max_length, contexts_length + question_length))
    
    # Context to context attention
    temp_attention_mask[:contexts_length, :contexts_length] = 1
    # Question to question attention
    temp_attention_mask[contexts_length : contexts_length + question_length,
                        contexts_length : contexts_length + question_length] = 1
    
    # CLS to every token
    temp_attention_mask[0, :contexts_length + question_length] = 1
    # Every token to CLS
    temp_attention_mask[:, 0] = 1
    if temp_attention_mask.shape[0] > max_seq_length:
        temp_attention_mask = temp_attention_mask[:max_seq_length, :max_seq_length]
    return temp_attention_mask.unsqueeze(0).detach()
    
    
def get_cross_lingual_attention_mask(contexts_length, question_length, max_seq_length, cross_alpha=0):
    max_length = max(max_seq_length, contexts_length + question_length)
    temp_attention_mask = torch.zeros(max_length, max_length)
    
    temp_attention_mask[:, :contexts_length + question_length] = torch.bernoulli(
        cross_alpha * torch.ones(max_length, contexts_length + question_length))
    
    # Context to question attention 
    temp_attention_mask[:contexts_length,
                        contexts_length: contexts_length + question_length] = 1
    # Question to context attention
    temp_attention_mask[contexts_length: contexts_length + question_length,
                        :contexts_length] = 1
    
    # CLS to every token
    temp_attention_mask[0, :contexts_length + question_length] = 1
    # Every token to CLS
    temp_attention_mask[:, 0] = 1
    if temp_attention_mask.shape[0] > max_seq_length:
        temp_attention_mask = temp_attention_mask[:max_seq_length, :max_seq_length]
    return temp_attention_mask.unsqueeze(0).detach()

class EuroparlPretrainingDataset(torch.utils.data.Dataset):
    """ A Dataset class to load Europarl Dataset for pre-training cross-lingual query. """
    def __init__(self, dataset, tokenizer, padding, max_seq_length, mono_alpha, cross_alpha, truncation=True,
                 code_switched_format=None):
        """ Initializes the dataset object.

        Args:
            dataset: Europarl dataset in the first language.
            tokenizer (transformers.AutoTokenizer): The tokenizer to tokenize the text.
            padding (bool/str): Whether to pad the sequence or not.
            max_seq_length: Maximum sequence length.
            mono_alpha: Probability of attention between two tokens from the same language using the monolingual Query.
            cross_alpha: Probability of attention between two tokens from different languages using the cross-lingual Query.
            truncation: Whether to truncate the sequences or not.
            code_switched_format: Format of the code-switched data (en-x, x-en, or x-x).
            overwrite_cache (bool, optional): Wether overwrite the data cache file. Defaults to False.
        """
            
        start = time.time()
        lt = time.time()
        self.encodings1 = []
        self.encodings2 = []
        
        self.max_seq_length = max_seq_length
        self.mono_alpha = mono_alpha
        self.cross_alpha = cross_alpha

        text1 = []
        text2 = []

        logger.info(f"Dataset size: {len(dataset)}")
        lang1, lang2 = dataset.config_name.split("-")

        for index, example in enumerate(dataset):
            t1 = example["translation"][lang1]
            t2 = example["translation"][lang2]
            text1.append(t1)
            text2.append(t2)
            if index % 50000 == 0 and index > 0:
                logger.info(f"Processed {index}({(index * 100.0) / len(dataset):.2f}%) examples in {(time.time() - lt):.2f} seconds.")
                lt = time.time()
                self.encodings1.append(tokenizer(text1,  padding=padding, max_length=max_seq_length,
                                                 return_tensors='pt', truncation=truncation))
                self.encodings2.append(tokenizer(text2,  padding=padding, max_length=max_seq_length,
                                                 return_tensors='pt', truncation=truncation))
                text1 = []
                text2 = []

        if len(text1) > 0:
            self.encodings1.append(tokenizer(text1,  padding=padding, max_length=max_seq_length,
                                             return_tensors='pt', truncation=truncation))
            self.encodings2.append(tokenizer(text2,  padding=padding, max_length=max_seq_length,
                                             return_tensors='pt', truncation=truncation))
            
        self.encodings1 = {key: torch.cat([enc[key] for enc in self.encodings1], dim=0)
                           for key, _ in self.encodings1[0].items()}
        self.encodings2 = {key: torch.cat([enc[key] for enc in self.encodings2], dim=0)
                           for key, _ in self.encodings2[0].items()}
        
        if code_switched_format == "en-x":
            if lang2 == "en":
                self.encodings1, self.encodings2 = self.encodings2, self.encodings1
                
        elif code_switched_format == "x-en":
            if lang1 == "en":
                self.encodings1, self.encodings2 = self.encodings2, self.encodings1
        elif code_switched_format == "x-x":
            temp1 = {key: [] for key in self.encodings1}
            temp2 = {key: [] for key in self.encodings1}
            for i in range(len(self.encodings1["input_ids"])):
                rand_index = random.randint(0, 1)
                if rand_index == 0:
                    for key in self.encodings1:
                        temp1[key].append(self.encodings1[key][i])
                        temp2[key].append(self.encodings2[key][i])
                else:
                    for key in self.encodings1:
                        temp2[key].append(self.encodings1[key][i])
                        temp1[key].append(self.encodings2[key][i])
            
            self.encodings1 = temp1
            self.encodings2 = temp2
                        
        elif code_switched_format is None:
            pass
        else:
            ValueError("Invalid code_switched_format")

        dataset_size = len(self.encodings1["input_ids"])
        logger.info(f"Number of samples: {dataset_size}, Time: {time.time() - start} seconds.")
            
    def __getitem__(self, idx):
        input_ids_context = self.encodings1['input_ids'][idx]
        attention_mask_context = self.encodings1['attention_mask'][idx]
        
        input_ids_question = self.encodings2['input_ids'][idx]
        attention_mask_question = self.encodings2['attention_mask'][idx]
        
        question_lengths = attention_mask_question.sum(dim=0) - 1
        contexts_lengths = attention_mask_context.sum(dim=0)

        input_ids = torch.cat([input_ids_context[:contexts_lengths], input_ids_question[1:question_lengths+1], 
                               torch.tensor([0] * (self.max_seq_length - (contexts_lengths + question_lengths)))])
        
        mono_attention_mask = get_mono_attention_mask(contexts_lengths, question_lengths, self.max_seq_length, self.mono_alpha)
        cross_attention_mask = get_cross_lingual_attention_mask(contexts_lengths, question_lengths, self.max_seq_length, self.cross_alpha)
        
        attention_mask = torch.cat([mono_attention_mask, cross_attention_mask], dim=0)
        
        if input_ids.shape[0] > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
        
        token_type_ids = torch.zeros(input_ids.shape[0]).long()
        item = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
        return item
    
    def __len__(self):
        return len(self.encodings1["input_ids"])
    
def load_pretraining_dataset(language_pair, dataset_name, tokenizer, args,
                             padding, truncation=True, overwrite_cache=False):
    """ Loads the pre-training dataset.

    Args:
        language_pair (str): Language pair for the dataset.
        dataset_name (str): Name of the dataset.
        tokenizer (transformers.AutoTokenizer): The tokenizer to tokenize the text.
        args (argparse.Namespace): Running arguments.
        padding (bool/str): Whether to pad the sequence or not.
        truncation: Whether to truncate the sequences or not.
        overwrite_cache (bool, optional): Wether overwrite the data cache file. Defaults to False.

    Returns:
        torch.utils.data.Dataset: The pre-training dataset.
    """
    lang1, lang2 = language_pair.split("-")
    
    if dataset_name == "xnli":
        d1 = load_dataset("xnli", lang1, split=f'train[:{args.train_data_percentage}%]')
        d2 = load_dataset("xnli", lang2, split=f'train[:{args.train_data_percentage}%]')
        data_class = XNLIPretrainingDataset
        train_dataset = data_class(language_pair, d1, d2, tokenizer, padding, args.max_seq_length,
                                   args.train_alpha, args.cross_alpha,
                                   is_parallel=args.is_parallel, code_switched_format=args.code_switched_format,
                                   truncation=truncation, overwrite_cache=overwrite_cache)
        
        d1 = load_dataset("xnli", lang1, split='validation')
        d2 = load_dataset("xnli", lang2, split='validation')
            
        eval_dataset = data_class(language_pair, d1, d2, tokenizer, padding, args.max_seq_length, args.train_alpha,
                                  args.cross_alpha, is_parallel=args.is_parallel,
                                  code_switched_format=args.code_switched_format, truncation=truncation,
                                  overwrite_cache=overwrite_cache)
    
    elif dataset_name == "europal":
        dataset = load_dataset("europarl_bilingual",
                               lang1=lang1,
                               lang2=lang2,
                               split=f'train[:{args.train_data_percentage}%]')
        split_dataset = dataset.train_test_split(test_size=args.val_data_ratio)
        t_dataset = split_dataset["train"]
        e_dataset = split_dataset["test"]
        
        train_dataset = EuroparlPretrainingDataset(t_dataset, tokenizer, padding, args.max_seq_length,
                                                  args.train_alpha, args.cross_alpha,
                                                  code_switched_format=args.code_switched_format)
        eval_dataset = EuroparlPretrainingDataset(e_dataset, tokenizer, padding, args.max_seq_length,
                                                 args.mono_eval_alpha, args.cross_eval_alpha,
                                                 code_switched_format=args.code_switched_format)
        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    return {"train": train_dataset, "eval": eval_dataset}

class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """
    def to(self, device):
        return self


class DataLoaderWithLangName:
    """
    Wrapper for a DataLoader to yield data in a language
    """

    def __init__(self, lang_name, data_loader):
        self.lang_name = lang_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        while True:
            for batch in self.data_loader:
                batch["lang_name"] = StrIgnoreDevice(f'{self.lang_name}')
                yield batch


class MultiDatasetDataloader:
    """
    Data loader that combines and samples from multiple languages.
    data loaders.
    """

    def __init__(self, dataloader_dict, evaluation=False, sampling_strategy=None, temperature=None):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(len(dataloader.dataset) for dataloader in self.dataloader_dict.values())
        self.sampling_strategy = sampling_strategy
        self.temperature = temperature
        self.evaluation = evaluation

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a language, and yield a batch from the respective
        language Dataloader.
        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        sampled_batch_numbers = self.size_proportional_sampling(self.num_batches_dict)
        if self.evaluation:
            self.sampling_strategy = 'no_sampling'
        if self.sampling_strategy == 'temperature':
            sampled_batch_numbers = self.temperature_sampling(self.num_batches_dict)
        elif self.sampling_strategy == 'size_proportional':
            sampled_batch_numbers = self.size_proportional_sampling(self.num_batches_dict)
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * sampled_batch_numbers[task_name]
        if sum(self.num_batches_dict.values()) - len(task_choice_list) > 0:
            random_tasks = random.choices(task_choice_list,
                                          k=sum(self.num_batches_dict.values()) - len(task_choice_list))
            for t in random_tasks:
                sampled_batch_numbers[self.task_name_list[t]] += 1
            task_choice_list += random_tasks
        task_choice_list = np.array(task_choice_list)
        if not self.evaluation:
            np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])

    def temperature_sampling(self, num_batches_dict):
        total_size = sum(num_batches_dict.values())
        sampling_ratios = {task_name: (size / total_size) ** (1.0 / self.temperature)
                           for task_name, size in num_batches_dict.items()}
        sampling_ratios = {task_name: sampling_ratios[task_name] / sum(sampling_ratios.values())
                           for task_name in num_batches_dict.keys()}
        sampled_numbers = {task_name: int(sampling_ratios[task_name] * sum(num_batches_dict.values()))
                           for task_name in num_batches_dict.keys()}
        return sampled_numbers

    def size_proportional_sampling(self, num_batches_dict):
        return num_batches_dict


def get_single_dataloader(lang_name, dataset, batch_size, data_collator, args):
    """
    Create a single-task data loader that also yields task names
    """
    train_sampler = (
        RandomSampler(dataset)
        if args.local_rank == -1
        else DistributedSampler(dataset)
    )

    data_loader = DataLoaderWithLangName(
        lang_name=lang_name,
        data_loader=DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            num_workers=1,
        ),
    )
    return data_loader

def get_ruletaker_dataset_sentences(data_folder):
    """ Prepares unique sentences for ruletaker datasets and saves it in a file.

    Args:
        data_folder (str): Path to the data folder.
    """
    start = time.time()
    dataset_folder = os.path.join(data_folder, "ruletaker/en/original")
    output_file_path = os.path.join(data_folder, "ruletaker/en/original/sentences.json")
    sentence_dict = {}
    folders = [d for d in os.listdir(dataset_folder) if not d.startswith(".") and os.path.isdir(os.path.join(dataset_folder, d))]
    
    for folder in folders:
        logger.info(f"******************* Processing folder: {folder} *******************\n")
        files = [f for f in os.listdir(os.path.join(dataset_folder, folder)) if f.endswith(".jsonl")]
        for file_path in files:
            if file_path.startswith("meta"):
                continue
            with open(os.path.join(dataset_folder, folder, file_path), 'r', encoding='utf-8') as reader:
                logger.info("==========================================================")
                logger.info(f"Processing file: {file_path}")
                for index, line in enumerate(reader):
                    sample = json.loads(line)
                    context = sample['context']
                    sentences =  nltk.sent_tokenize(context)
                    for sent in sentences:
                        if sent not in sentence_dict:
                            sentence_dict[sent] = 0
                        sentence_dict[sent] += 1                    
                    
                    for q in sample["questions"]:
                        if q["text"] not in sentence_dict:
                            sentence_dict[q["text"]] = 0
                        sentence_dict[q["text"]] += 1

                    if index % 1000 == 0:
                        logger.info(f" {index} samples processed.")

        logger.info(f"Folder {folder} processed in {time.time() - start} seconds.\n")
        start = time.time()
        
    logger.info(f"{len(sentence_dict)} unique sentences are found.")
    logger.info("Saving the sentence dictionary...")
    
    with open(output_file_path, "w") as write_file:
        json.dump(sentence_dict, write_file, indent=2)
    logger.info("Saving is done.")


def get_LeapOfThought_dataset_sentences(data_folder):
    """ Prepares unique sentences for ruletaker datasets and saves it in a file.

    Args:
        data_folder (str): Path to the data folder.
    """
    start = time.time()
    output_file_path = os.path.join(data_folder, "sentences.json")
    sentence_dict = {}
    files = [d for d in os.listdir(data_folder)
             if not d.startswith(".")
             and os.path.isfile(os.path.join(data_folder, d))
             and not d.startswith("concept_lang_map")
             and not d.startswith("sentences")
             and not d.startswith("artificial")]
    
    for file_path in files:
        with open(os.path.join(data_folder, file_path), 'r', encoding='utf-8') as reader:
            logger.info("==========================================================")
            logger.info(f"Processing file: {file_path}")
            for index, line in enumerate(reader):
                sample = json.loads(line)
                
                if sample["phrase"] not in sentence_dict:
                    sentence_dict[sample["phrase"]] = 0
                sentence_dict[sample["phrase"]] += 1
                
                sample = sample["metadata"]
                sentences = sample["rules"]
                for sent in sentences:
                    if sent not in sentence_dict:
                        sentence_dict[sent] = 0
                    sentence_dict[sent] += 1                    

                if index % 1000 == 0:
                    logger.info(f" {index} samples processed.")

    logger.info(f"Dataset processed in {time.time() - start} seconds.\n")
    start = time.time()
        
    logger.info(f"{len(sentence_dict)} unique sentences are found.")
    logger.info("Saving the sentence dictionary...")
    
    with open(output_file_path, "w") as write_file:
        json.dump(sentence_dict, write_file, indent=2)
    logger.info("Saving is done.")


def translate_sentences(input_file, src_lang, dest_lang):
    """ Translates sentences from one language to another and saves it in a file.

    Args:
        input_file (str): Path to the input file (pickle file).
        src_lang (str): Source language.
        dest_lang (str): Target language.
    """
    logger.info(f"Translating sentences from {src_lang} to {dest_lang}...")
    translator = Translator()

    translation_dict = {}
    directory, _ = os.path.split(input_file)
    output_file_path = os.path.join(directory, f"sentences_{src_lang}_to_{dest_lang}_translated.json")
    
    with open(input_file, "r", encoding='utf-8') as handle:
        sentence_dict = json.load(handle)
        sentences = list(sentence_dict.keys())
    
    if os.path.exists(output_file_path):
        translation_dict = json.load(open(output_file_path, "r", encoding='utf-8'))
        sentences = sentences[len(translation_dict):]
        logger.info(f"{len(translation_dict)} translated sentences are loaded.")

    logger.info(f"Translating {len(sentences)} sentences...")
    block_size = 1000
    start = time.time()
    t0 = time.time()
    i = 0
    while i < len(sentences):
        if i + block_size > len(sentences) - 1:
            break
        try:
            block_trans = translator.translate(sentences[i:i+block_size], dest=dest_lang)
            for j in range(i, i + len(block_trans)):
                translation_dict[sentences[j]] = block_trans[j - i].text
                    
            logger.info(f"Block {i}: Translation time: {time.time() - t0}")
            t0 = time.time()

            if i % 1000 == 0 and i != 0:
                logger.info(f"{i} sentences translated in {time.time() - start} seconds.")
                logger.info("==========================================================")
                with open(output_file_path, "w", encoding='utf-8') as writer:
                    json.dump(translation_dict, writer, indent=2, ensure_ascii=False)
                
            i += block_size
               
        except:
            logger.warning("Time out error. Waiting for 10 seconds...")
            time.sleep(10)
            translator = Translator()
            
    last_block_trans = translator.translate(sentences[i:len(sentences)], dest=dest_lang)

    for j in range(i, i + len(last_block_trans)):
        translation_dict[sentences[j]] = last_block_trans[j - i].text
    
    logger.info(f"Translation is done in {time.time() - start} seconds.")
    logger.info("Saving the sentence dictionary...")
        
    with open(output_file_path, "w", encoding='utf-8') as writer:
        json.dump(translation_dict, writer, indent=2,  ensure_ascii=False)
    logger.info("Saving is done.")


def leapOfthought_data_from_translated_sent(data_folder, dest_lang):
    """ Prepares LeapOfThought datasets from translated sentences.
    
    Args:
        data_folder (str): Path to the data folder.
        src_lang (str): Source language.
        dest_lang (str): Target language.
    """
    sentence_file_path = os.path.join(data_folder, f"sentences_en_to_{dest_lang}_translated.json")
    with open(sentence_file_path, "r", encoding='utf-8') as handle:
        translation_dict = json.load(handle)
    
    dataset_folder = os.path.join(data_folder, "en")
    output_folder = os.path.join(data_folder, f"{dest_lang}")
    os.makedirs(output_folder,  exist_ok=True)

    files = [d for d in os.listdir(dataset_folder) if d.endswith("jsonl") and d.startswith("randomized_hypernyms_")]

    for file_path in files:
        data = []
        with open(os.path.join(dataset_folder, file_path), 'r', encoding='utf-8') as reader:
            logger.info("==========================================================")
            logger.info(f"Processing file: {file_path}")
            t0 = time.time()
            for index, line in enumerate(reader):
                
                sample = json.loads(line)
                
                sample["phrase"] = translation_dict[sample["phrase"]]
                sentences = sample["metadata"]["rules"]
                translated_sents = [translation_dict[sent].strip() for sent in sentences]
                for i in range(len(translated_sents)):
                    if not translated_sents[i].endswith("."):
                        translated_sents[i] += "."
                    
                translated_context = " ".join(translated_sents)
                sample["context"] = translated_context
                data.append(sample)
                
                if index % 10000 == 0:
                    logger.info(f"Processed {index} samples.")
        
        with open(os.path.join(output_folder, file_path), 'w', encoding='utf-8') as writer:
            for entry in data:
                json.dump(entry, writer, ensure_ascii=False)
                writer.write('\n')
        logger.info(f"Processing is done for file {file_path} in {time.time() - t0} seconds.\n")
        logger.info("==========================================================\n")


def ruletaker_data_from_translated_sent(data_folder, src_lang, dest_lang):
    """ Prepares ruletaker datasets from translated sentences.
    
    Args:
        data_folder (str): Path to the data folder.
        src_lang (str): Source language.
        dest_lang (str): Target language.
    """
    start = time.time()
    sentence_file_path = os.path.join(data_folder,
                                      f"ruletaker/en/original/sentences_{src_lang}_to_{dest_lang}_translated.json")
    with open(sentence_file_path, "r", encoding='utf-8') as handle:
        translation_dict = json.load(handle)
    
    dataset_folder = os.path.join(data_folder, "ruletaker/en/original")
    output_folder = os.path.join(data_folder, f"ruletaker/{dest_lang}/original")
    os.makedirs(output_folder,  exist_ok=True)
    
    folders = [d for d in os.listdir(dataset_folder)
               if not d.startswith(".")
               and os.path.isdir(os.path.join(dataset_folder, d))]
    for folder in folders:
        logger.info(f"******************* Processing folder: {folder} *******************\n")
        os.makedirs(os.path.join(output_folder, folder), exist_ok=True)
        files = [f for f in os.listdir(os.path.join(dataset_folder, folder)) if f.endswith(".jsonl")]

        for file_path in files:
            if file_path.startswith("meta"):
                continue
            data = []
            with open(os.path.join(dataset_folder, folder, file_path), 'r', encoding='utf-8') as reader:
                logger.info("==========================================================")
                logger.info(f"Processing file: {file_path}")
                t0 = time.time()
                for index, line in enumerate(reader):
                    sample = json.loads(line)
                    context = sample['context']
                    sentences =  nltk.sent_tokenize(context)
                    translated_sents = [translation_dict[sent].strip() for sent in sentences]
                    for i in range(len(translated_sents)):
                        if not translated_sents[i].endswith("."):
                            translated_sents[i] += "."
                    translated_context = " ".join(translated_sents)
                    sample["context"] = translated_context
                    
                    questions = []
                    for q in sample["questions"]:
                        q["text"] = translation_dict[q["text"]]
                        if not q["text"].strip().endswith("."):
                            q["text"] += "."
                        questions.append(q)
                    sample["questions"] = questions
                    data.append(sample)
                    
                    if index % 1000 == 0:
                        logger.info(f"Processed {index} samples.")
            
            with open(os.path.join(output_folder, folder, file_path), 'w', encoding='utf-8') as writer:
                for entry in data:
                    json.dump(entry, writer, ensure_ascii=False)
                    writer.write('\n')
            logger.info(f"Processing is done for file {file_path} in {time.time() - t0} seconds.\n")
            logger.info("==========================================================\n")
        logger.info(f"Folder {folder} processed in {time.time() - start} seconds.\n")

