import os
import sys
import shutil

import fire
import torch
import transformers
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import time

import os.path as osp
from typing import Union

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

fetch_time = 0.0
forward_time = 0.0

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        self.template = dict()
        self.template["description"] = "Template used by Alpaca-LoRA."
        self.template["prompt_input"] = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        self.template["prompt_no_input"] = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        self.template["response_split"] = "### Response:"
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


def main(
    load_8bit: bool = False,
    base_model: str = "decapoda-research/llama-7b-hf",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
):
    total_start = time.time_ns()
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)

    global fetch_time
    global forward_time
    start = time.time_ns()
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=False,
    )

    fetch_time += (time.time_ns() - start) / 1e9

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        max_new_tokens=40,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        # beam search
        # generation_config = GenerationConfig(
        #     pad_token_id=tokenizer.eos_token_id,
        #     num_beams=num_beams,
        #     **kwargs,
        # )

        # sampling
        generation_config = GenerationConfig(
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs,
        )

        # disable kv cache
        generation_config.use_cache=False

        global fetch_time
        global forward_time
        start = time.time_ns()
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        forward_time += (time.time_ns() - start) / 1e9

        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield prompter.get_response(output)


    """
    # testing code for readme
    """
    sample_start = time.time_ns()

    for instruction in [
        "Tell me about alpacas.",
        # "Tell me about the president of Mexico in 2019.",
        # "Tell me about the king of France in 2019.",
        # "List all Canadian provinces in alphabetical order.",
        # "Write a Python program that prints the first 10 Fibonacci numbers.",
        # "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        # "Tell me five words that rhyme with 'shock'.",
        # "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        # "Count up from 1 to 500.",
    ]:
        print("Questions:", instruction)
        sentence = ""
        for tok in evaluate(instruction):
            sentence = sentence + " " + tok
        print(f"Response: {sentence}")

    print(f"skip_layer / total_layer (%): {model.model.skip_layer} / {model.model.total_layer} \
          ({model.model.skip_layer / model.model.total_layer * 100: .2f}%)")
    print(f"fetch_time {fetch_time}")
    print(f"forward_time {forward_time}")
    print(f"Total sample time = {(time.time_ns() - sample_start) / 1e9}")
    print(f"Total total time = {(time.time_ns() - total_start) / 1e9}")


if __name__ == "__main__":
    fire.Fire(main)
