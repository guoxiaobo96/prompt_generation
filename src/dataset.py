"""Dataset utils for different data settings for GLUE."""

import os
import copy
import logging
from re import search
import torch
import numpy as np
import time
from filelock import FileLock
import json
import itertools
import random
import transformers
from transformers.data.processors.utils import InputFeatures
from transformers import DataProcessor, InputExample
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer, util
from copy import deepcopy
import pandas as pd


import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers.tokenization_utils_base import BatchEncoding


InputDataClass = NewType("InputDataClass", Any)


from processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, median_mapping
from config import MiscArgument, DynamicDataTrainingArguments, GeneratorArgument, MLModelArguments, TrainingArguments, get_config

logger = logging.getLogger(__name__)



@dataclass(frozen=True)
class OurSingleInputFeatures(InputFeatures):
    """
    Inherit from Transformers' InputFeatuers.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    mask_pos: Optional[List[int]] = None # Position of the mask token
    label_word_list: Optional[List[int]] = None # Label word mapping (dynamic)

    # support_list:List[OurInputFeatures] = None

@dataclass(frozen=True)
class OurInputFeatures(InputFeatures):
    """
    Inherit from Transformers' InputFeatuers.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    mask_pos: Optional[List[int]] = None # Position of the mask token
    label_word_list: Optional[List[int]] = None # Label word mapping (dynamic)
    sentence: Optional[List[str]] = None
    prompts: Optional[str] = None

    support_list:List[OurSingleInputFeatures] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

def input_example_to_string(example, sep_token): 
    if example.text_b is None:
        return example.text_a
    else:
        # Warning: very simple hack here
        return example.text_a + ' ' + sep_token + ' ' + example.text_b

def input_example_to_tuple(example): 
    if example.text_b is None:
        if pd.isna(example.text_a) or example.text_a is None:
            return ['']
            logger.warn("Empty input")
        else:
            return [example.text_a]
    else:
        return [example.text_a, example.text_b]

def tokenize_multipart_input(
    input_text_list, 
    max_length, 
    tokenizer, 
    first_sent_limit=None,
    truncate_head=False,
):
    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    input_ids = []
    attention_mask = []
    token_type_ids = [] # Only for BERT
    mask_pos = None # Position of the mask token


    input_ids = [tokenizer.cls_token_id]
    attention_mask = [1]
    token_type_ids = [0]

    for sent_id, input_text in enumerate(input_text_list):
        if input_text is None:
            # Do not have text_b
            continue
        if pd.isna(input_text) or input_text is None:
            # Empty input
            input_text = ''
        input_tokens = enc(input_text) + [tokenizer.sep_token_id]
        input_ids += input_tokens
        attention_mask += [1 for i in range(len(input_tokens))]
        token_type_ids += [sent_id for i in range(len(input_tokens))]

    # Padding
    if first_sent_limit is not None and len(input_ids) > max_length:
        # If using sentence limit, the total length still exceeds the maximum limit, report a warning
        logger.warn("Input exceeds max_length limit: {}".format(tokenizer.decode(input_ids)))

    while len(input_ids) < max_length:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)
        token_type_ids.append(0)

    # Truncate
    if len(input_ids) > max_length:
        if truncate_head:
            input_ids = input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]
            token_type_ids = token_type_ids[-max_length:]
        else:
            # Default is to truncate the tail
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]

    # Find mask token
    mask_pos = [input_ids.index(tokenizer.mask_token_id)]
    # Make sure that the masked position is inside the max_length
    assert mask_pos[0] < max_length

    result = {'input_ids': input_ids, 'attention_mask': attention_mask}
    if 'BERT' in type(tokenizer).__name__:
        # Only provide token type ids for BERT
        result['token_type_ids'] = token_type_ids

    result['mask_pos'] = mask_pos

    return result



class FewShotDataset(torch.utils.data.Dataset):
    """Few-shot dataset."""

    def __init__(self, misc_args,  args, tokenizer, filter_model = None, filter_trainer = None, cache_dir=None, mode="train", dynamic_prompt=True, dynamic_label=True):
        self.misc_args = misc_args
        self.args = args
        self.task_name = args.task_name
        self.processor = processors_mapping[args.task_name]
        self.tokenizer = tokenizer
        self.mode = mode
        self.filter_model = filter_model
        if self.filter_model is not None:
            self.trainer = filter_trainer

        # If not using demonstrations, use dynamic_prompt=True
        self.dynamic_prompt = dynamic_prompt
        self.dynamic_label = dynamic_label

        if self.dynamic_prompt:
            logger.info("Use dynamic_prompt")
        if self.dynamic_label:
            logger.info("Use dynamic_label")

        assert mode in ["train", "dev", "test"]

        # Get label list and (for prompt) label word list
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)


        # Load cache
        # Cache name distinguishes mode, task name, tokenizer, and length. So if you change anything beyond these elements, make sure to clear your cache.
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                args.task_name,
            ),
        )

        logger.info(f"Creating/loading examples from dataset file at {args.data_dir}")

        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.support_examples, self.query_examples = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                # The support examples are sourced from the training set.
                self.support_examples = self.processor.get_train_examples(args.data_dir)

                if mode == "dev":
                    self.query_examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == "test":
                    self.query_examples = self.processor.get_test_examples(args.data_dir)
                else:
                    self.query_examples = self.support_examples

                start = time.time()
                torch.save([self.support_examples, self.query_examples], cached_features_file)

                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )



        if not self.dynamic_label:
            assert args.mapping is not None
            self.label_to_word = eval(args.mapping)

            for key in self.label_to_word:
                # For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
                if self.label_to_word[key][0] not in ['<', '[', '.', ',']:
                    # Make sure space+word is in the vocabulary
                    assert len(tokenizer.tokenize(' ' + self.label_to_word[key])) == 1
                    self.label_to_word[key] = tokenizer._convert_token_to_id(tokenizer.tokenize(' ' + self.label_to_word[key])[0])
                else:
                    self.label_to_word[key] = tokenizer._convert_token_to_id(self.label_to_word[key])
                logger.info("Label {} to word {} ({})".format(key, tokenizer._convert_id_to_token(self.label_to_word[key]), self.label_to_word[key]))
            self.label_word_list = [self.label_to_word[label] for label in self.label_list]
        else:
            self.label_to_word = None
            self.label_word_list = None

        if not self.dynamic_prompt:
            assert args.template is not None
            self.prompts_list = [[args.template] for _ in self.size]
        else:
            self.prompts_list = dict()
            prompt_file = os.path.join(args.data_dir,mode+'_prompt')
            with open(prompt_file,mode='r',encoding='utf8') as fp:
                for line in fp.readlines():
                    item = json.loads(line.strip())
                    self.prompts_list[item['guid']] = item
        
        # If it is not training, we pre-process the data; otherwise, we process the data online.
        self.features = []
        if self.misc_args.global_debug:
            self.query_examples = self.query_examples[:4]

        for _, example in enumerate(self.query_examples):
            prompts = self.prompts_list[example.guid]
            
            supports = self.support_examples
            self.features.extend(self.convert_fn(
                example=example,
                supports=supports,
                dynamic_prompt=self.dynamic_prompt,
                label_list=self.label_list,
                prompts=prompts,
                tokenizer = self.tokenizer,
                label_word_list=self.label_word_list,
                verbose=True if _ == 0 else False,
            ))
        self.size = len(self.features)



    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.features is None:
            query_idx, context_indices, bootstrap_idx = self.example_idx[i]
            # The input (query) example
            example = self.query_examples[query_idx]
            # The demonstrations
            # supports = self.select_context([self.support_examples[i] for i in context_indices])
            supports = [self.support_examples[i] for i in context_indices]

            if self.args.template_list is not None:
                template = self.args.template_list[i % len(self.args.template_list)]
            else:
                template = self.args.template

            features = self.convert_fn(
                example=example,
                supports=supports,
                dynamic_prompt=self.dynamic_prompt,
                label_list=self.label_list,
                prompt=self.args.prompt,
                prompts=template,
                label_word_list=self.label_word_list,
                verbose=False,
            )
        else:
            features = self.features[i]
            
        return features

    def get_labels(self):
        return self.label_list


    def convert_fn(
        self,
        example,
        supports,
        dynamic_prompt=True,
        label_list=None,
        prompts=None,
        tokenizer = None,
        label_word_list=None,
        verbose=False
    ):
        """
        Returns a list of processed "InputFeatures".
        """
        max_length = self.args.max_seq_length    

        # Prepare labels
        label_map = {label: i for i, label in enumerate(label_list)} # Mapping the label names to label ids
        if len(label_list) == 1:
            # Regression
            label_map = {'0': 0, '1': 1}

        # Get example's label id (for training/inference)
        if example.label is None:
            example_label = None
        elif len(label_list) == 1:
            # Regerssion
            example_label = float(example.label)
        else:
            example_label = label_map[example.label]


        example_list = list()
        # Prepare other features
        if dynamic_prompt:
            if example.text_b is None:
                for i, prompt in enumerate(prompts['gen_text_a']):
                    single_example = list()
                    # prompt_list = prompt.split(' ')
                    prompt_list = ['']
                    for index,_ in enumerate(prompt_list):
                        new_prompt_list = prompt_list[:index] + [tokenizer.mask_token] +prompt_list[index+1:]
                        new_prompt = ' '.join(new_prompt_list)
                        new_prompt = new_prompt.replace('  ',' ')
                        example_temp  = copy.deepcopy(example)
                        example_temp.text_a += new_prompt
                        supports_temp = list()
                        for item in supports:
                            item_temp = copy.deepcopy(item)
                            item_temp.text_a += new_prompt
                            supports_temp.append(item_temp)
                        single_example.append((example_temp,supports_temp,[example.text_a], new_prompt))
                    if self.filter_model is not None:
                        single_example = self.filter_prompt(single_example)
                    
                    example_list.extend(single_example)

        features_list = list()
        for example in example_list:
            # No using dynamic prompts
            query_input = tokenize_multipart_input(
                input_text_list=input_example_to_tuple(example[0]),
                max_length=max_length,
                tokenizer=self.tokenizer,
                first_sent_limit=self.args.first_sent_limit,
            )
            support_list = list()
            for support in example[1]:
                support_input = tokenize_multipart_input(
                    input_text_list=input_example_to_tuple(support),
                    max_length=max_length,
                    tokenizer=self.tokenizer,
                    first_sent_limit=self.args.first_sent_limit,
                )
                support_list.append(OurSingleInputFeatures(**support_input,label=label_map[support.label]))
            features = OurInputFeatures(**query_input, support_list = support_list, label=example_label,sentence=example[2],prompts=example[3])
            features_list.append(features)
        # if verbose:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("features: %s" % features)
        #     logger.info("text: %s" % self.tokenizer.decode(features.input_ids))

        return features_list

    def filter_prompt(self, example_list):
        res_list = list()
        for item in example_list:
            feature = self.filter_model.tokenizer(item[0].text_a,padding='max_length',max_length = self.args.max_seq_length,return_tensors='pt')
            for k, v in feature.items():
                feature[k] = v.to(self.filter_model.device)
            pred = self.filter_model(**feature)
            pred = np.argmax(pred.logits.detach().cpu()[0])
            if pred == 1:
                res_list.append(item)
        return res_list

def few_shot_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    import torch

    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids","support_list",'sentence','prompts') and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])
    if "support_list" in first:
        batch["support_list"] = [few_shot_data_collator(f["support_list"]) for f in features]
    if "sentence" in first:
        batch["sentence"] = [f["sentence"] for f in features]
    if "prompts" in first:
        batch["prompts"] = [f["prompts"] for f in features]

    return batch

class FilterDataset(torch.utils.data.Dataset):
    def __init__(self, misc_args, args, tokenizer, cache_dir=None, mode="train"):
        self.misc_args = misc_args
        self.args = args
        self.task_name = args.task_name
        self.processor = processors_mapping[args.task_name]
        self.tokenizer = tokenizer
        self.mode = mode

        # If not using demonstrations, use dynamic_prompt=True
        assert mode in ["train", "dev", "test"]

        # Load cache
        # Cache name distinguishes mode, task name, tokenizer, and length. So if you change anything beyond these elements, make sure to clear your cache.
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}_{}".format(
                mode,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                args.task_name,
                misc_args.task.split('_')[-1]
            ),
        )

        logger.info(f"Creating/loading examples from dataset file at {args.data_dir}")

        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.query_examples = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")
                file = os.path.join(args.data_dir,mode+'_score')
                index = 0
                examples = []
                with open(file, mode='r',encoding='utf8') as fp:
                    for line in fp.readlines():
                        item = json.loads(line.strip())
                        guid = "%s-%s" % (mode, index)
                        text_a = tokenizer(item['text'],padding='max_length',max_length = self.args.max_seq_length)
                        label = int(item['label'])
                        examples.append(InputFeatures(**text_a, label=label))
                self.query_examples = examples

                # The support examples are sourced from the training set.

                start = time.time()
                torch.save(self.query_examples, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )


        self.features = self.query_examples
        self.size = len(self.features)

    def __len__(self):
        return self.size

    def _get_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def __getitem__(self, i):
        features = self.features[i]
        return features

class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, cache_dir=None, mode="train"):
        self.args = args
        self.task_name = args.task_name
        self.processor = processors_mapping[args.task_name]
        self.tokenizer = tokenizer
        self.mode = mode

        # If not using demonstrations, use dynamic_prompt=True
        assert mode in ["train", "dev", "test"]

        # Load cache
        # Cache name distinguishes mode, task name, tokenizer, and length. So if you change anything beyond these elements, make sure to clear your cache.
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                args.task_name,
            ),
        )

        logger.info(f"Creating/loading examples from dataset file at {args.data_dir}")

        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.support_examples, self.query_examples = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                # The support examples are sourced from the training set.
                self.support_examples = self.processor.get_train_examples(args.data_dir)

                if mode == "dev":
                    self.query_examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == "test":
                    self.query_examples = self.processor.get_test_examples(args.data_dir)
                else:
                    self.query_examples = self.support_examples

                start = time.time()
                torch.save([self.support_examples, self.query_examples], cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )


        self.features = self.query_examples

    def __len__(self):
        return len(self.features)


    def __getitem__(self, i):
        features = self.features[i]
        return features


def main():
    from transformers import AutoTokenizer
    misc_args, data_args, generator_args, ml_model_args, training_args = get_config()
    tokenizer = AutoTokenizer.from_pretrained(
        ml_model_args.ml_tokenizer_name if ml_model_args.ml_tokenizer_name else ml_model_args.ml_model_name_or_path,
        additional_special_tokens=[],
        cache_dir=ml_model_args.ml_cache_dir,
    )
    # dataset = FewShotDataset(data_args,tokenizer)
    dataset = FilterDataset(misc_args,data_args, tokenizer)
    

if __name__ == '__main__':
    main()