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


from src.config import  get_config
from src.run_function import generate_prompts, generate_score, train_filter, train_ml

from filelock import FileLock
from datetime import datetime

from copy import deepcopy
from tqdm import tqdm
import json
import torch
import random
import numpy as np

logger = logging.getLogger(__name__)



def main():
    misc_args, data_args, generator_args, filter_model_args, ml_model_args, training_args = get_config()
    set_seed(misc_args.global_seed)
    if misc_args.task == "generate_prompt":
        generate_prompts(misc_args,data_args,generator_args)
    elif misc_args.task == "generate_score":
        generate_score(misc_args,data_args,generator_args,ml_model_args,training_args)
    elif misc_args.task == "train_filter":
        train_filter(misc_args,data_args,generator_args,filter_model_args,training_args)
    elif misc_args.task == "train_model":
        train_ml(misc_args, data_args, generator_args, filter_model_args, ml_model_args,training_args)



if __name__ == "__main__":
    main()