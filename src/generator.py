import argparse
import logging

import numpy as np
import torch

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

from config import MiscArgument, GeneratorArgument, get_config
# from src.config import MiscArgument, GeneratorArgument, get_config



class Generator(object):
    def __init__(self, misc_args:MiscArgument, generator_args:GeneratorArgument) -> None:
        super().__init__()
        self._PREPROCESSING_FUNCTIONS = {"ctrl": self._prepare_ctrl_input, "xlm": self._prepare_xlm_input,
                                         "xlnet": self._prepare_xlnet_input, "transfo-xl": self._prepare_transfoxl_input}
        self._MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

        self._MODEL_CLASSES = {
            "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
            "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
            "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
            "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
            "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
            "xlm": (XLMWithLMHeadModel, XLMTokenizer),
        }

        self._PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
        (except for Alexei and Maria) are discovered.
        The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
        remainder of the story. 1883 Western Siberia,
        a young Grigori Rasputin is asked by his father and a group of men to perform magic.
        Rasputin has a vision and denounces one of the men as a horse thief. Although his
        father initially slaps him for making such an accusation, Rasputin watches as the
        man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
        the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
        with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self._logger = logging.getLogger(__name__)

        self._misc_args = misc_args
        self._generator_args = generator_args

        self._prepare()


    def _prepare(self):
        # Initialize the model and tokenizer
        try:
            self._generator_args.generator_model_type = self._generator_args.generator_model_type.lower()
            model_class, tokenizer_class = self._MODEL_CLASSES[self._generator_args.generator_model_type]
        except KeyError:
            raise KeyError(
                "the model {} you specified is not supported. You are welcome to add it and open a PR :)")

        self._tokenizer = tokenizer_class.from_pretrained(self._generator_args.generator_model_name_or_path)
        self._model = model_class.from_pretrained(self._generator_args.generator_model_name_or_path)
        self._model.to(self._generator_args.generator_device)

        if self._generator_args.generator_fp16:
            self._model.half()
        self._generator_args.generator_length = self._adjust_length_to_model(self._generator_args.generator_length, max_sequence_length=self._model.config.max_position_embeddings)
        self._requires_preprocessing = self._generator_args.generator_model_type in self._PREPROCESSING_FUNCTIONS.keys()

    def generate(self, prompt_text):
        if self._requires_preprocessing:
            prepare_input = self._PREPROCESSING_FUNCTIONS.get(self._generator_args.generator_model_type)
            preprocessed_prompt_text = prepare_input(self._generator_args,self._model, self._tokenizer, prompt_text)

            if self._model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
                tokenizer_kwargs = {"add_space_before_punct_symbol": True}
            else:
                tokenizer_kwargs = {}

            encoded_prompt = self._tokenizer.encode(
                preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
            )
        else:
            prefix = self._generator_args.generator_prefix
            encoded_prompt = self._tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self._generator_args.generator_device)
        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        output_sequences = self._model.generate(
            input_ids=input_ids,
            max_length=self._generator_args.generator_length + len(encoded_prompt[0]),
            temperature=self._generator_args.generator_temperature,
            top_k=self._generator_args.generator_k,
            top_p=self._generator_args.generator_p,
            repetition_penalty=self._generator_args.generator_repetition_penalty,
            do_sample=True,
            num_return_sequences=self._generator_args.generator_num_return_sequences,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = self._tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            text = text[: text.find(self._generator_args.generator_stop_token) if self._generator_args.generator_stop_token else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                prompt_text + text[len(self._tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
            )

            generated_sequences.append(total_sequence)
            print(total_sequence)

        return generated_sequences

    def _prepare_ctrl_input(self,args:GeneratorArgument, _, tokenizer, prompt_text):
        if args.generator_temperature > 0.7:
            self._logger.info(
                "CTRL typically works better with lower temperatures (and lower top_k).")

        encoded_prompt = tokenizer.encode(
            prompt_text, add_special_tokens=False)
        if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
            self._logger.info(
                "WARNING! You are not starting your generation from a control code so you won't get good results")
        return prompt_text

    def _prepare_xlm_input(self,args:GeneratorArgument, model, tokenizer, prompt_text):
        # kwargs = {"language": None, "mask_token_id": None}

        # Set the language
        use_lang_emb = hasattr(
            model.config, "use_lang_emb") and model.config.use_lang_emb
        if hasattr(model.config, "lang2id") and use_lang_emb:
            available_languages = model.config.lang2id.keys()
            if args.generator_xlm_language in available_languages:
                language = args.generator_xlm_language
            else:
                language = None
                while language not in available_languages:
                    language = input(
                        "Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

            model.config.lang_id = model.config.lang2id[language]
            # kwargs["language"] = tokenizer.lang2id[language]

        # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
        # XLM masked-language modeling (MLM) models need masked token
        # is_xlm_mlm = "mlm" in args.model_name_or_path
        # if is_xlm_mlm:
        #     kwargs["mask_token_id"] = tokenizer.mask_token_id

        return prompt_text

    def _prepare_xlnet_input(self,args:GeneratorArgument, _, tokenizer, prompt_text):
        prefix = args.generator_prefix if args.generator_prefix else self._PREFIX
        prompt_text = prefix + prompt_text
        return prompt_text

    def _prepare_transfoxl_input(self,args:GeneratorArgument, _, tokenizer, prompt_text):
        prefix = args.generator_prefix if args.generator_prefix else self._PREFIX
        prompt_text = prefix + prompt_text
        return prompt_text

    def _adjust_length_to_model(self,length, max_sequence_length):
        if length < 0 and max_sequence_length > 0:
            length = max_sequence_length
        elif 0 < max_sequence_length < length:
            length = max_sequence_length  # No generation bigger than model size
        elif length < 0:
            length = self._MAX_LENGTH  # avoid infinite loop
        return length



def test():
    misc_args, generator_args = get_config()
    generator = Generator(misc_args, generator_args)
    generator.generate("this is an amazing book")

if __name__ == "__main__":
    test()
