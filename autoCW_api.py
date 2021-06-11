#!/usr/bin/env python
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import sys
import os
import yaml
import argparse
import datetime
import json
import time
import warnings
import glob
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.seq2seq.utils import (
    calculate_bleu, calculate_rouge, chunks, parse_numeric_n_bool_cl_kwargs, use_task_specific_params
)


logger = getLogger(__name__)


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_summaries_or_translations(
    examples: List[str],
    out_file: str,
    model_name: str,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    fp16=False,
    task="summarization",
    prefix=None,
    **generate_kwargs,
) -> Dict:
    """Save model.generate results to <out_file>, and return how long it took."""
    model_name = str(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if fp16:
        model = model.half()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Inferred tokenizer type: {tokenizer.__class__}")  # if this is wrong, check config.model_type.

    start_time = time.time()
    # update config with task specific params
    use_task_specific_params(model, task)
    if prefix is None:
        prefix = prefix or getattr(model.config, "prefix", "") or ""
    for examples_chunk in tqdm(list(chunks(examples, batch_size))):
        examples_chunk = [prefix + text for text in examples_chunk]
        batch = tokenizer(examples_chunk, return_tensors="pt", truncation=True, padding="longest").to(device)
        summaries = model.generate(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            **generate_kwargs,
        )
        dec = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    runtime = int(time.time() - start_time)  # seconds
    n_obs = len(examples)
    return dict(n_obs=n_obs, runtime=runtime, seconds_per_sample=round(runtime / n_obs, 4)), dec


def datetime_now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_args():
    """ Load args and run some basic checks.
        Args loaded from:
        - Huggingface transformers training args (defaults for using their model)
        - Manual args from .yaml file
    """
    with open(f'config/seq2seq/call_api.yaml', 'r') as f:
        args = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))
    return args


def run_generate(args, parsed_args, examples, verbose=True):
    """
    Takes input text, generates output, and then using reference calculates the BLEU scores.

    The results are saved to a file and returned to the caller, and printed out unless ``verbose=False`` is passed.

    Args:
        verbose (:obj:`bool`, `optional`, defaults to :obj:`True`): print results to stdout

    Returns:
        a tuple: ``(scores, params}``
        - ``scores``: a dict of scores data ``{'bleu': 39.6501, 'n_obs': 2000, 'runtime': 186, 'seconds_per_sample': 0.093}``
        - ``params``: a dict of custom params, e.g. ``{'num_beams': 5, 'length_penalty': 0.8}``
    """
    if args.n_obs > 0:
        examples = examples[: args.n_obs]
        
    _, generations = generate_summaries_or_translations(
        examples,
        args.save_path,
        args.model_name,
        batch_size=args.bs,
        device=args.device,
        fp16=args.fp16,
        task=args.task,
        prefix=args.prefix,
        **parsed_args,
    )

    return generations


def main(json_data):
    args = load_args()
    parsed_args = parse_numeric_n_bool_cl_kwargs([])

    # Set empty args
    args.input_path = None
    args.reference_path = None
    args.save_path = None
    args.score_path = None

    section_outputs = {}
    for sec_num, examples in json_data.items():
        sec_args = argparse.Namespace(**vars(args))
        section = f'section_{sec_num}'
        sec_args.model_name = os.path.join(sec_args.models_dir, section)

        generations = run_generate(sec_args, parsed_args, examples, verbose=True)
        section_outputs[section] = generations

    return section_outputs


def file_to_json():
    # Load data and format as json
    dirs = glob.glob('data/go/section_*')
    json_data = {}
    for folder in dirs:
        sec_num = int(folder[-1])
        assert sec_num in range(1,8)
        path = os.path.join(folder, 'test.source')
        examples = [x.strip() for x in open(path).readlines()]
        json_data[sec_num] = examples
    return json_data


if __name__ == "__main__":
    json_data = file_to_json()
    main(json_data)