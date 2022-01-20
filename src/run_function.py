"""Finetuning the library models for sequence classification on GLUE."""

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import transformers

from transformers.trainer import Trainer
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers import BertForSequenceClassification, RobertaForSequenceClassification
from transformers.utils.dummy_pt_objects import RobertaModel

from .dataset import FewShotDataset, PromptDataset, few_shot_data_collator, FilterDataset
from .ml_models import BertForPromptFinetuning, RobertaForPromptFinetuning, resize_token_type_embeddings, filter_metrics, res_metrics, acc_metrics
from .processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, bound_mapping
from .config import MiscArgument, DynamicDataTrainingArguments, GeneratorArgument, FilterModelArguments, MLModelArguments, TrainingArguments, get_config
from .generator import Generator
from .trainer import DynamicTrainer

from filelock import FileLock
from datetime import datetime

from copy import deepcopy
from tqdm import tqdm
import json
import torch
import random
import numpy as np

logger = logging.getLogger(__name__)



def generate_prompts(misc_args: MiscArgument, data_args: DynamicDataTrainingArguments, generator_args: GeneratorArgument):
    generator = Generator(misc_args, generator_args)
    for mode in ['train', 'dev','test']:
        dataset = PromptDataset(data_args, tokenizer=generator.tokenizer, mode=mode)
        data = list()
        for item in tqdm(dataset,total=len(dataset)):
            generated_sentence_list = generator.generate(item.text_a)
            generated_sentence_list = [item[1] for item in generated_sentence_list]
            record = {'guid':item.guid, 'text_a':item.text_a, 'gen_text_a':generated_sentence_list,'text_b':item.text_b}
            if item.text_b is not None:
                generated_sentence_list = generator.generate(item.text_b)
                generated_sentence_list = [item[1] for item in generated_sentence_list]
                record['gen_text_b'] = generated_sentence_list
            else:
                record['gen_text_b'] = list()
            record['label'] = item.label
            data.append(record)
        prompt_file = os.path.join(data_args.data_dir,mode+'_prompt')
        with open(prompt_file,mode='w',encoding='utf8') as fp:
            for item in data:
                fp.write(json.dumps(item,ensure_ascii=False)+'\n')
        

def generate_score(misc_args: MiscArgument, data_args: DynamicDataTrainingArguments, generator_args: GeneratorArgument, ml_model_args: MLModelArguments, training_args: TrainingArguments):
    # Create config
    num_labels = num_labels_mapping[data_args.task_name]
    config = AutoConfig.from_pretrained(
        ml_model_args.ml_config_name if ml_model_args.ml_config_name else ml_model_args.ml_model_name_or_path,
        num_labels=num_labels,
        cache_dir=ml_model_args.ml_cache_dir,
    )

    if config.model_type == 'roberta':
        model_fn = RobertaForPromptFinetuning
    elif config.model_type == 'bert':
        model_fn = BertForPromptFinetuning
    else:
        raise NotImplementedError
    special_tokens = []

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        ml_model_args.ml_tokenizer_name if ml_model_args.ml_tokenizer_name else ml_model_args.ml_model_name_or_path,
        additional_special_tokens=special_tokens,
        cache_dir=ml_model_args.ml_cache_dir,
    )

    model = model_fn.from_pretrained(
        ml_model_args.ml_model_name_or_path,
        from_tf=bool(".ckpt" in ml_model_args.ml_model_name_or_path),
        config=config,
        cache_dir=ml_model_args.ml_cache_dir,
    )

    # For BERT, increase the size of the segment (token type) embeddings
    if config.model_type == 'bert':
        model.resize_token_embeddings(len(tokenizer))
        resize_token_type_embeddings(model, new_num_types=10, random_segment=ml_model_args.ml_random_segment)

    # Pass dataset and argument information to the model
    model.model_args = ml_model_args
    model.data_args = data_args
    model.tokenizer = tokenizer
    model.vocab = list(tokenizer.get_vocab())
    model.return_full_softmax = True



    # for mode in ['train','dev']:
    for mode in ['test']:
        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=None,
            data_collator = few_shot_data_collator
        )
    # First we compute zero-shot logits on all of the examples.
        dataset = FewShotDataset(misc_args, data_args, tokenizer=tokenizer, mode=mode)

        # Predict logits.
        dataloader = trainer.get_eval_dataloader(dataset)
        output = trainer.prediction_loop(dataloader, description="Evaluation")
        logits = output.predictions[0] if isinstance(output.predictions, (list, tuple)) else output.predictions
        labels = output.label_ids
        record_list = list()
        for i, item in enumerate(dataset):
            if i >= len(dataset):
                break
            text = item.sentence[0] + item.prompts
            if len(item.sentence) > 1:
                text += item.sentence[1]
            text = text.replace('  ',' ')
            score = '0'
            if logits[i][0] == logits[i][1]:
                score = '0'
            else:
                pred_label = np.argmax(logits[i])
                if pred_label == labels[i]:
                    score = '1'
            record_list.append({'text':text,'label':score})
        record_file = os.path.join(data_args.data_dir, mode+'_score')
        with open(record_file,mode='w',encoding='utf8') as fp:
            for item in record_list:
                fp.write(json.dumps(item,ensure_ascii=False)+'\n')
    return
    


def train_filter(misc_args: MiscArgument, data_args: DynamicDataTrainingArguments, generator_args: GeneratorArgument, filter_model_args: FilterModelArguments, training_args: TrainingArguments):
    # Create config
    num_labels = num_labels_mapping[data_args.task_name]
    config = AutoConfig.from_pretrained(
        filter_model_args.filter_config_name if filter_model_args.filter_config_name else filter_model_args.filter_model_name_or_path,
        num_labels=num_labels,
        cache_dir=filter_model_args.filter_cache_dir,
    )

    if config.model_type == 'roberta':
        model_fn = RobertaForSequenceClassification
    elif config.model_type == 'bert':
        model_fn = BertForSequenceClassification
    else:
        raise NotImplementedError
    special_tokens = []

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        filter_model_args.filter_tokenizer_name if filter_model_args.filter_tokenizer_name else filter_model_args.filter_model_name_or_path,
        additional_special_tokens=special_tokens,
        cache_dir=filter_model_args.filter_cache_dir,
    )

    model = model_fn.from_pretrained(
        filter_model_args.filter_model_name_or_path,
        from_tf=bool(".ckpt" in filter_model_args.filter_model_name_or_path),
        config=config,
        cache_dir=filter_model_args.filter_cache_dir,
    )

    # For BERT, increase the size of the segment (token type) embeddings
    if config.model_type == 'bert':
        model.resize_token_embeddings(len(tokenizer))
        resize_token_type_embeddings(model, new_num_types=10, random_segment=filter_model_args.filter_random_segment)

    # Pass dataset and argument information to the model
    model.model_args = filter_model_args
    model.data_args = data_args
    model.tokenizer = tokenizer
    model.vocab = list(tokenizer.get_vocab())

    # Initialize our Trainer

    train_dataset = FilterDataset(misc_args, data_args, tokenizer=tokenizer, mode='train')
    eval_dataset = FilterDataset(misc_args, data_args, tokenizer=tokenizer, mode='dev')
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=filter_metrics
    )
    trainer.train()
    eval_res = trainer.evaluate()
    print(eval_res)
    if trainer.is_world_process_zero():
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        torch.save(filter_model_args, os.path.join(training_args.output_dir, "model_args.bin"))
        torch.save(data_args, os.path.join(training_args.output_dir, "data_args.bin"))
        torch.save(training_args, os.path.join(training_args.output_dir, "training_args.bin"))
    return
    

def train_ml(misc_args: MiscArgument, data_args: DynamicDataTrainingArguments, generator_args: GeneratorArgument, filter_model_args: FilterModelArguments, ml_model_args: MLModelArguments, training_args: TrainingArguments):
    num_labels = num_labels_mapping[data_args.task_name]
    ml_config = AutoConfig.from_pretrained(
        ml_model_args.ml_config_name if ml_model_args.ml_config_name else ml_model_args.ml_model_name_or_path,
        num_labels=num_labels,
        cache_dir=ml_model_args.ml_cache_dir,
    )

    if ml_config.model_type == 'roberta':
        model_fn = RobertaForPromptFinetuning
    elif ml_config.model_type == 'bert':
        model_fn = BertForPromptFinetuning
    else:
        raise NotImplementedError
    special_tokens = []
    # Create tokenizer
    ml_tokenizer = AutoTokenizer.from_pretrained(
        ml_model_args.ml_tokenizer_name if ml_model_args.ml_tokenizer_name else ml_model_args.ml_model_name_or_path,
        additional_special_tokens=special_tokens,
        cache_dir=ml_model_args.ml_cache_dir,
    )

    ml_model = model_fn.from_pretrained(
        ml_model_args.ml_model_name_or_path,
        from_tf=bool(".ckpt" in ml_model_args.ml_model_name_or_path),
        config=ml_config,
        cache_dir=ml_model_args.ml_cache_dir,
    )

    # For BERT, increase the size of the segment (token type) embeddings
    if ml_config.model_type == 'bert':
        ml_model.resize_token_embeddings(len(ml_tokenizer))
        resize_token_type_embeddings(ml_model, new_num_types=10, random_segment=ml_model_args.ml_random_segment)

    # Pass dataset and argument information to the model
    ml_model.model_args = ml_model_args
    ml_model.data_args = data_args
    ml_model.tokenizer = ml_tokenizer
    ml_model.vocab = list(ml_tokenizer.get_vocab())



    filter_training_args = torch.load(os.path.join(filter_model_args.filter_model_name_or_path, "training_args.bin"))  
    data_args = torch.load(os.path.join(filter_model_args.filter_model_name_or_path, "data_args.bin"))  
    filter_model_name_or_path = filter_model_args.filter_model_name_or_path
    filter_model_args = torch.load(os.path.join(filter_model_args.filter_model_name_or_path, "model_args.bin"))
    filter_model_args.filter_model_name_or_path = filter_model_name_or_path

    filter_model_config = AutoConfig.from_pretrained(
        filter_model_args.filter_config_name if filter_model_args.filter_config_name else filter_model_args.filter_model_name_or_path,
        num_labels=num_labels,
        cache_dir=filter_model_args.filter_cache_dir,
    )
    filter_tokenizer = AutoTokenizer.from_pretrained(
        filter_model_args.filter_tokenizer_name if filter_model_args.filter_tokenizer_name else filter_model_args.filter_model_name_or_path,
        additional_special_tokens=special_tokens,
        cache_dir=filter_model_args.filter_cache_dir,
    )



    if filter_model_config.model_type == 'roberta':
        filter_model_fn = RobertaForSequenceClassification
    elif filter_model_config.model_type == 'bert':
        filter_model_fn = BertForSequenceClassification
    else:
        raise NotImplementedError
    special_tokens = []

    filter_model = filter_model_fn.from_pretrained(filter_model_args.filter_model_name_or_path)
    filter_model = filter_model.to(training_args.device)



    filter_model.model_args = filter_model_args
    filter_model.data_args = data_args
    filter_model.tokenizer = filter_tokenizer


    
    filter_model.eval()
    # Initialize our Trainer

    # train_dataset = FewShotDataset(misc_args, data_args, filter_model=filter_model,tokenizer=ml_tokenizer, mode='train')
    # eval_dataset = FewShotDataset(misc_args, data_args, filter_model=filter_model,tokenizer=ml_tokenizer, mode='dev')
    # test_dataset = FewShotDataset(misc_args, data_args, filter_model=filter_model,tokenizer=ml_tokenizer, mode='test')

    train_dataset = FewShotDataset(misc_args, data_args, tokenizer=ml_tokenizer, mode='train')
    eval_dataset = FewShotDataset(misc_args, data_args, tokenizer=ml_tokenizer, mode='dev')
    test_dataset = FewShotDataset(misc_args, data_args, tokenizer=ml_tokenizer, mode='test')

    # trainer = Trainer(
    #     model=ml_model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     compute_metrics = res_metrics,
    #     data_collator = few_shot_data_collator
    # )
    trainer = DynamicTrainer(
        model=ml_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics = acc_metrics,
        data_collator = few_shot_data_collator
    )
    trainer.train()
    eval_res = trainer.evaluate()
    print(eval_res)

    test_res = trainer.evaluate(eval_dataset = test_dataset)
    print(test_res)
    return

def main():
    misc_args, data_args, generator_args, filter_model_args, ml_model_args, training_args = get_config()
    set_seed(misc_args.global_seed)
    # generate_prompts(misc_args,data_args,generator_args)
    # generate_score(misc_args,data_args,generator_args,ml_model_args,training_args)
    train_filter(misc_args,data_args,generator_args,filter_model_args,training_args)
    # train_ml(misc_args, data_args, generator_args, filter_model_args, ml_model_args,training_args)



if __name__ == "__main__":
    main()