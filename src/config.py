import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, NewType
from typing_extensions import Required

from transformers import MODEL_WITH_LM_HEAD_MAPPING, HfArgumentParser, set_seed, logging, TrainingArguments, GlueDataTrainingArguments, training_args
import transformers
from transformers.trainer_utils import default_compute_objective
import torch
import sys

@dataclass
class MiscArgument:
    """
    Arguments pertrain to the misc arguments about the run environment of the program
    """
    task: str = field(
        metadata={"help": "The task of running"}
    )
    root_dir: str = field(
        default='/home/xiaobo/prompt_generation', metadata={"help": "The absolute path of the root dir"}
    )
    log_dir: Optional[str] = field(
        default='/data/xiaobo/prompt_generation/log', metadata={"help": "The absolute path to the log dir"}
    )

    global_debug: bool = field(
        default=False, metadata={"help": "Whether the program is in debug mode"}
    )

    global_seed: int = field(
        default=42, metadata={
            "help": "Random seed that will be set at the beginning of training."}
    )


@dataclass
class GeneratorArgument:
    generator_model_type: str = field(
        default=None,
        metadata={"help": "Model type selected in the list"}
    )

    generator_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pre-trained model or shortcut name"}
    )
    generator_length: int = field(
        default=20
    )
    generator_stop_token: str = field(
        default=None,
        metadata={"help": "Token at which text generation is stopped"}
    )
    generator_temperature: float = field(
        default=1.0,
        metadata={
            "help": "temperature of 1.0 has no effect, lower tend toward greedy sampling"}
    )
    generator_repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "primarily useful for CTRL model; in that case, use 1.2"}
    )
    generator_k: int = field(
        default=0
    )
    generator_p: float = field(
        default=0.9
    )
    generator_prefix: str = field(
        default="",
        metadata={"help": "Text added prior to input."}
    )
    generator_xlm_language: str = field(
        default="",
        metadata={"help": "Optional language when used with the XLM model."}
    )

    generator_num_return_sequences: int = field(
        default=1,
        metadata={"help": "The number of samples to generate."}
    )

    generator_fp16: bool = field(
        default=False,
        metadata={
            "help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"}
    )

    generator_n_gpu: int = field(
        default=0,
        metadata={"help": "The number of gpu for generating"}
    )
    generator_device: torch.device = field(
        default=None,
        metadata={"help": "The device for generator"}
    )


@dataclass
class MLModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    ml_model_name_or_path: Optional[str] = field(
        default=None, metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    ml_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    ml_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    ml_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    # Few-shot type
    #   - finetune: standard fine-tuning
    #   - prompt: prompt-based fine-tuning
    #   - prompt-demo: prompt-based fine-tuning with demonstrations
    ml_few_shot_type: str = field(
        default='prompt-demo',
        metadata={
            "help": "Few-shot learning model type. Choice: finetune, prompt, prompt-demo"}
    )

    # Only for BERT-type model
    ml_random_segment: bool = field(
        default=False,
        metadata={
            "help": "Whether to reinitialize the token type embeddings (only for BERT)."}
    )


@dataclass
class FilterModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    filter_model_name_or_path: Optional[str] = field(
        default=None, metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    filter_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    filter_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    filter_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    # Few-shot type
    #   - finetune: standard fine-tuning
    #   - prompt: prompt-based fine-tuning
    #   - prompt-demo: prompt-based fine-tuning with demonstrations


    # Only for BERT-type model
    filter_random_segment: bool = field(
        default=False,
        metadata={
            "help": "Whether to reinitialize the token type embeddings (only for BERT)."}
    )


@dataclass
class DynamicDataTrainingArguments(GlueDataTrainingArguments):
    """
    Arguments for dynamic training.
    """
    num_k: Optional[int] = field(
        default=16,
        metadata={"help": "Number of training instances per class"}
    )

    num_sample: Optional[int] = field(
        default=16,
        metadata={
            "help": "Number of samples (for inference) in fine-tuning with demonstrations"}
    )

    num_demo: Optional[int] = field(
        default=1,
        metadata={"help": "Number of demonstrations from each class"}
    )

    auto_demo: bool = field(
        default=True,
        metadata={
            "help": "Automatically generate template for using demonstrations"}
    )

    # For prompting
    template: str = field(
        default=None,
        metadata={"help": "Template"}
    )

    mapping: str = field(
        default=None,
        metadata={"help": "Label word mapping"}
    )

    template_path: str = field(
        default=None,
        metadata={
            "help": "Path to a txt file that stores all the templates, one per line. Do not set this when prompt_path is used"}
    )

    mapping_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the label word mappings, one per line. Do not set this when prompt_path is used"}
    )

    prompt_path: str = field(
        default=None,
        metadata={
            "help": "Path to a txt file that stores all the prompts (templates and mappings), one per line"}
    )

    template_id: int = field(
        default=None,
        metadata={"help": "Template id if using template_path"}
    )

    mapping_id: int = field(
        default=None,
        metadata={"help": "Mapping id if using template_path"}
    )

    prompt_id: int = field(
        default=None,
        metadata={"help": "Prompt id if using prompt_path"}
    )

    top_n_template: int = field(
        default=None,
        metadata={"help": "Use top-n template in the template path"}
    )

    # For logging
    tag: str = field(
        default='',
        metadata={"help": "Set the tag and find the result easier in the log."}
    )

    # For filtering when using demonstrations
    demo_filter: bool = field(
        default=False,
        metadata={"help": "Only use similar instances in demonstrations"}
    )

    demo_filter_rate: float = field(
        default=0.5,
        metadata={"help": "Only use top-x\% similar instances in demonstrations"}
    )

    demo_filter_model: str = field(
        default=None,
        metadata={
            "help": "Model name for demonstration filter embeddings. Will load embeddings based on the model name."}
    )

    debug_mode: bool = field(
        default=False,
        metadata={"help": "Debug mode"}
    )

    # For max length
    double_demo: bool = field(
        default=False,
        metadata={"help": "Use double length for using demonstrations"}
    )

    first_sent_limit: int = field(
        default=None,
        metadata={
            "help": "Limit the length of the first sentence (i.e., sent_0)"}
    )

    other_sent_limit: int = field(
        default=None,
        metadata={
            "help": "Limit the length of sentences other than the first sentence"}
    )

    use_full_length: bool = field(
        default=None,
        metadata={"help": "Use the full length (512)"}
    )



    # # GPT-3's in-context learning
    # gpt3_in_context_head: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "GPT-3's in-context learning (context at the beginning)"}
    # )

    # gpt3_in_context_tail: bool = field(
    #     default=False,
    #     metadata={"help": "GPT-3's in-context learning (context at the end)"}
    # )

    # gpt3_in_context_num: int = field(
    #     default=32,
    #     metadata={"help": "Number of context examples"}
    # )

    truncate_head: bool = field(
        default=False,
        metadata={
            "help": "When exceeding the maximum length, truncate the head instead of the tail."}
    )

    # Do not set up the following fields. They are set up automatically.
    prompt: bool = field(
        default=False,
        metadata={"help": "Whether to use prompt-based fine-tuning"}
    )
    template_list: list = field(
        default=None,
        metadata={
            "help": "(DO NOT List of templates (only initialized after the program starts."}
    )


@dataclass
class DynamicTrainingArguments(TrainingArguments):
    # For ensemble
    array_id: int = field(
        default=-1,
        metadata={
            "help": "Array ID (contains seed and hyper-paramter search) to idenfity the model"}
    )

    model_id: int = field(
        default=-1,
        metadata={
            "help": "Model ID (contains template information) to identify the model"}
    )

    save_logit: bool = field(
        default=False,
        metadata={
            "help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"}
    )

    save_logit_dir: str = field(
        default=None,
        metadata={"help": "Where to save the prediction result"}
    )

    # Regularization
    fix_layers: int = field(
        default=0,
        metadata={"help": "Fix bottom-n layers when optimizing"}
    )

    # Training
    save_at_last: bool = field(
        default=False,
        metadata={
            "help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"}
    )

    # Turn off train/test
    no_train: bool = field(
        default=False,
        metadata={"help": "No training"}
    )
    no_predict: bool = field(
        default=False,
        metadata={"help": "No test"}
    )

    update_epoch: int = field(
        default=None,
        metadata={
            "help": "The epoch for updating the predicton model"}
    )

def get_config() -> Tuple:

    def _get_config(misc_ars: MiscArgument, data_args: DynamicDataTrainingArguments, generator_args: GeneratorArgument, filter_model_args: FilterModelArguments, ml_model_args: MLModelArguments, trainging_args: TrainingArguments):
        generator_args.generator_n_gpu = 0 if trainging_args.no_cuda else torch.cuda.device_count()
        generator_args.generator_device = torch.device(
            "cuda" if torch.cuda.is_available() and not trainging_args.no_cuda else "cpu")
        training_args.seed = misc_args.global_seed
        if 'prompt' in ml_model_args.ml_few_shot_type:
            data_args.prompt = True
        if training_args.no_train:
            training_args.do_train = False
        if training_args.no_predict:
            training_args.do_predict = False
        data_args.data_dir = os.path.join(data_args.data_dir,str(data_args.num_k)+'-'+str(misc_ars.global_seed))

    parser = HfArgumentParser((MiscArgument, DynamicDataTrainingArguments,
                              GeneratorArgument, FilterModelArguments, MLModelArguments, DynamicTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        misc_args, data_args, generator_args, ml_model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        misc_args, data_args, generator_args, filter_model_args, ml_model_args, training_args = parser.parse_args_into_dataclasses()
    _get_config(misc_args, data_args, generator_args,filter_model_args,
                ml_model_args, training_args)
    return misc_args, data_args, generator_args, filter_model_args, ml_model_args, training_args


def test():
    misc_args, data_args, generator_args, filter_model_args, ml_model_args, training_args = get_config()
    print('test finish')


if __name__ == '__main__':
    test()
