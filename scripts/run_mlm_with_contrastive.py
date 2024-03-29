#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# copy from example/language_modeling/run_mlm.py
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

#from datasets import load_dataset
from datasets import concatenate_datasets, load_dataset, DatasetDict
from datasets import load_from_disk

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.6.0.dev0")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


from transformers import Trainer
from torch.nn import CrossEntropyLoss
from sampler import postive_random_sampler, negative_random_sampler
import torch.nn.functional as F
import torch

class TrainerWithContrastiveLoss(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.pop("labels")
        outputs = model(output_hidden_states=True, **inputs)
        logits = outputs.logits


        # MLM loss
        loss_fct = CrossEntropyLoss() # -100 for ignoring
        mlmloss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)) # (bs, seq_len, num_labels) -> (bs * seq_len, num_labels)


        # INFONCE loss
        embeddings = outputs.hidden_states[0]
        last_hiddens = outputs.hidden_states[-1]
        hidden_size = last_hiddens.size(-1)
        # sample pos and neg (get sample index)
        pos_sample_row_indices, pos_sample_col_indices = postive_random_sampler(labels, window_size=self.args.pos_window_size)
        neg_sample_row_indices, neg_sample_col_indices = negative_random_sampler(labels, sample_n=self.args.neg_k)
        batch_size, max_seq_len, pos_sample_n = pos_sample_row_indices.shape
        # get positive samples from this batch
        pos_sample_row_indices_ = pos_sample_row_indices.view(-1)
        pos_sample_col_indices_ = pos_sample_col_indices.view(-1)
        positives = last_hiddens[pos_sample_row_indices_, pos_sample_col_indices_, :] # shape (bs * seq_len * 1, hid)
        positives = positives.view(batch_size, max_seq_len, hidden_size)               # shape (bs, seq_len, hid)
        # get negative samples from this batch
        neg_sample_row_indices_ = neg_sample_row_indices.view(-1)
        neg_sample_col_indices_ = neg_sample_col_indices.view(-1)
        negatives = last_hiddens[neg_sample_row_indices_, neg_sample_col_indices_, :] # shape (bs * seq_len * sample_n, hid)
        negatives = negatives.view(batch_size, max_seq_len, -1, hidden_size)           # shape (bs, seq_len, sample_n, hid)

        # global semantic = hidden - embed
        if True:
            lambda_coeff = 1.0
            # anchor vectors
            last_hiddens -= lambda_coeff * embeddings
            # postive vectors
            positive_embeddings = embeddings[pos_sample_row_indices_, pos_sample_col_indices_, :]
            positive_embeddings = positive_embeddings.view(batch_size, max_seq_len, hidden_size)
            positives -= lambda_coeff * positive_embeddings 
            # negative vectors
            negative_embeddings = embeddings[neg_sample_row_indices_, neg_sample_col_indices_, :]
            negative_embeddings = negative_embeddings.view(batch_size, max_seq_len, -1, hidden_size)
            negatives -= lambda_coeff * negative_embeddings

        # generate mask, in mlm we have
        #    masked_indices = torch.bernoulli(probability_matrix).bool() # [MASK] -> True
        #    labels[~masked_indices] = -100  # We only compute loss on masked tokens # not [MASK] -> -100
        mask = (labels != -100) # [MASK]s
        if self.args.contrastive_for_all_pos:
            if self.args.contrastive_except_mlm_pos:
                # for_all_pos=True & except_mlm_pos=True -> 15% mlm + 85% tc
                tcloss = self.infonce_loss(last_hiddens, positives, negatives, ~mask) # not [MASK]s
            else:
                # for_all_pos=True & except_mlm_pos=False -> 15% mlm + 100% tc
                tcloss = self.infonce_loss(last_hiddens, positives, negatives)
        else:
            # for_all_pos=False -> 15% mlm + 15% tc
            tcloss = self.infonce_loss(last_hiddens, positives, negatives, mask)
        

        # OVERALL loss
        loss = mlmloss + tcloss * self.args.infonce_weight
        
        return (loss, outputs) if return_outputs else loss

    def infonce_loss(self, anchor, pos, neg, mask=None, tau=None):
        # see https://github.com/facebookresearch/moco/blob/master/moco/builder.py

        # normalization
        anchor = F.normalize(anchor, p=2, dim=-1)
        pos = F.normalize(pos, p=2, dim=-1).detach()
        neg = F.normalize(neg, p=2, dim=-1).detach()

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nlc,nlc->nl', [anchor, pos]).unsqueeze(-1)        # shape (bs, seq_len, 1)
        # negative logits: NxK
        l_neg = torch.einsum('nlc,nlkc->nlk', [anchor, neg])                    # shape (bs, seq_len, sample_n)

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=-1)                              # shape (bs, seq_len, 1+K)

        # apply temperature
        if tau is None:
            logits /= self.args.tau
        else:
            logits /= tau

        # labels: positive key indicators
        labels = torch.zeros((logits.shape[0], logits.shape[1]), dtype=torch.long).cuda() # shape (bs, seq_len)
        # Note that: all anchor, pos, neg are thought as "valid" ones 
        # Otherwise the "labels" should be set to -100 at some illegal position
        # e.g. in mlm
        #    masked_indices = torch.bernoulli(probability_matrix).bool()
        #    labels[~masked_indices] = -100  # We only compute loss on masked tokens
        if mask is not None: 
            labels = torch.where(mask, labels, -100)

        # dequeue and enqueue
        #self._dequeue_and_enqueue(k)
        #return logits, labels

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))      # (bs * seq_len, 1+K) and (bs * seq_len)
        return loss

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
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
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    tau: float = field(
        default=0.07, metadata={"help": "Temperature for contrastive loss"}
    )
    infonce_weight: float = field(
        default=1, metadata={"help": "loss weight for contrastive loss (in the form of infonce)"}
    )
    neg_k: int = field(
        default=50, metadata={"help": "how much negatives per anchor for building infonce loss"}
    )
    pos_window_size: int = field(
        default=5, metadata={"help": "the size of sampling window, for sampling 1 positive per anchor for building infonce loss"}
    )
    contrastive_for_all_pos: bool = field(
        default=True, metadata={"help": "infonce loss on all positions (True, as default) or only on the masked positions (False)"}
    )
    contrastive_except_mlm_pos: bool = field(
        default=False, metadata={"help": "infonce loss on positions except the masked ones or including, default as False (i.e. including)"}
    )
    

    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "If not None, load a preprocessed data from local disk."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=0,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )

    '''
    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."
    '''


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    '''
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    '''
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if data_args.dataset_path is not None and os.path.exists(data_args.dataset_path):
        # if processed data exists, load it from local disk
        tokenized_datasets = load_from_disk(data_args.dataset_path)
    else:
        # we don't need to specify the dataset due to the same setting with "bert": bookcopurs + wiki
        if data_args.validation_split_percentage == 0:
            print("[INFO] no validation split loaded")
            # load full datasets => return DatasetDict with corresponding keys, e.g. only 'train' here
            datasets = DatasetDict()
            bookcorpus = load_dataset("bookcorpus", split="train")
            wiki = load_dataset("wikipedia", "20220301.en", split="train")  # might be slightly diff with what used for bert
            wiki = wiki.remove_columns("title")  # only keep the text
            wiki = wiki.remove_columns("id")
            wiki = wiki.remove_columns("url")
            assert bookcorpus.features.type == wiki.features.type
            datasets["train"] = concatenate_datasets([bookcorpus, wiki])
        else:
            print(f"[INFO] {data_args.validation_split_percentage}% data as validation set")
            # load partial datasets => return Dataset, so we need to generate an DatasetDict to wrap them up
            # select the last validation_split_percentage% part text in bookcorpus & wiki for validation
            datasets = DatasetDict()
            bookcorpus_train = load_dataset("bookcorpus", split=f"train[:{100-data_args.validation_split_percentage}%]")
            bookcorpus_val = load_dataset("bookcorpus", split=f"train[-{data_args.validation_split_percentage}%:]")
            wiki_train = load_dataset("wikipedia", "20220301.en", split=f"train[:{100-data_args.validation_split_percentage}%]")
            wiki_train = wiki_train.remove_columns("title")
            wiki_train = wiki_train.remove_columns("id")
            wiki_train = wiki_train.remove_columns("url")
            wiki_val = load_dataset("wikipedia", "20220301.en", split=f"train[-{data_args.validation_split_percentage}%:]")
            wiki_val = wiki_val.remove_columns("title")
            wiki_val = wiki_val.remove_columns("id") 
            wiki_val = wiki_val.remove_columns("url")
            assert bookcorpus_train.features.type == bookcorpus_val.features.type
            assert bookcorpus_train.features.type == wiki_train.features.type
            assert bookcorpus_train.features.type == wiki_val.features.type
            datasets["train"] = concatenate_datasets([bookcorpus_train, wiki_train])
            datasets["validation"] = concatenate_datasets([bookcorpus_val, wiki_val])     
            #datasets["train"] = bookcorpus_train
            #datasets["validation"] = bookcorpus_val


        # truncate dataset first, then feed them to tokenizer
        if training_args.do_train:
            if "train" not in datasets:
                raise ValueError("--do_train requires a train dataset")
            if data_args.max_train_samples is not None:
                datasets["train"] = datasets["train"].select(range(data_args.max_train_samples)) #100000))
        if training_args.do_eval:
            if "validation" not in datasets:
                raise ValueError("--do_eval requires a validation dataset")
            if data_args.max_val_samples is not None:
                datasets["validation"] = datasets["validation"].select(range(data_args.max_val_samples)) #10000))

        # Preprocessing the datasets.
        # First we tokenize all the texts.
        if training_args.do_train:
            column_names = datasets["train"].column_names
        else:
            column_names = datasets["validation"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        if data_args.max_seq_length is None:
            max_seq_length = tokenizer.model_max_length
            if max_seq_length > 1024:
                logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
                )
                max_seq_length = 1024
        else:
            if data_args.max_seq_length > tokenizer.model_max_length:
                logger.warning(
                    f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                    f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                )
            max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        if data_args.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if data_args.pad_to_max_length else False

            def tokenize_function(examples):
                # Remove empty lines
                examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
                return tokenizer(
                    examples["text"],
                    padding=padding,
                    truncation=True,
                    max_length=max_seq_length,
                    # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                    # receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                )

            tokenized_datasets = datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not data_args.overwrite_cache,
            )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
            # efficient when it receives the `special_tokens_mask`.
            def tokenize_function(examples):
                return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

            tokenized_datasets = datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                total_length = (total_length // max_seq_length) * max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

        # save processed data to local disk, avoiding re-process
        if data_args.dataset_path is not None: # but not exists, save the processed data avoiding re-process later
            # it is a DatasetDict class (it will be put to the hdfs in the train.sh)
            tokenized_datasets.save_to_disk(data_args.dataset_path)

    if training_args.do_train:
        train_dataset = tokenized_datasets["train"]

    if training_args.do_eval:
        eval_dataset = tokenized_datasets["validation"]

    # Data collator
    # This one will take care of randomly masking the tokens.
    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
    )

    # Initialize our Trainer
    '''
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    '''
    training_args.tau = data_args.tau
    training_args.infonce_weight = data_args.infonce_weight
    training_args.neg_k = data_args.neg_k
    training_args.pos_window_size = data_args.pos_window_size
    training_args.contrastive_for_all_pos = data_args.contrastive_for_all_pos
    training_args.contrastive_except_mlm_pos = data_args.contrastive_except_mlm_pos
    trainer = TrainerWithContrastiveLoss(
        model=model,
        args=training_args, # which works as a proxy for passing hyper-params
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
        perplexity = math.exp(metrics["eval_loss"])
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
