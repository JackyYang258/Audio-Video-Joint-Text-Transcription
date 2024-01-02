# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ast
from itertools import chain
import logging
import math
import os
import sys
import json
import hashlib
import editdistance
from argparse import Namespace

import numpy as np
import torch
from fairseq import checkpoint_utils, options, tasks, utils, distributed_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.models import FairseqLanguageModel
from omegaconf import DictConfig

from pathlib import Path
import hydra
from hydra._internal.utils import get_args
from hydra.core.config_store import ConfigStore
from fairseq.dataclass.configs import (
    CheckpointConfig,
    CommonConfig,
    CommonEvalConfig,
    DatasetConfig,
    DistributedTrainingConfig,
    GenerationConfig,
    FairseqDataclass,
)
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from omegaconf import OmegaConf

from fairseq import search

from dataset import AudioVideoDataset
from models.audio_video_model import AudioVideoModelConfig, AudioVideoModel
from sequence_generator import SequenceGenerator
import utils
from utils.configs.arguments import TaskConfig

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OverrideConfig(FairseqDataclass):
    noise_wav: Optional[str] = field(default=None, metadata={'help': 'noise wav file'})
    noise_prob: float = field(default=0, metadata={'help': 'noise probability'})
    noise_snr: float = field(default=0, metadata={'help': 'noise SNR in audio'})
    modalities: List[str] = field(default_factory=lambda: [""], metadata={'help': 'which modality to use'})
    data: Optional[str] = field(default=None, metadata={'help': 'path to test data directory'})
    label_dir: Optional[str] = field(default=None, metadata={'help': 'path to test label directory'})


@dataclass
class InferConfig(FairseqDataclass):
    task: TaskConfig = TaskConfig()
    generation: GenerationConfig = GenerationConfig()
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    override: OverrideConfig = OverrideConfig()
    is_ax: bool = field(
        default=False,
        metadata={
            "help": "if true, assumes we are using ax for tuning and returns a tuple for ax to consume"
        },
    )


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos, generator.pad}


@hydra.main(config_path='configs', config_name="s2s_decode")
def main(cfg):
    # set up output dir
    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(cfg.common_eval.results_path, "decode.log")
    
    with open(output_path, "w", buffering=1, encoding="utf-8") as output_file:
        logging.basicConfig(
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=os.environ.get("LOGLEVEL", "INFO").upper(),
            stream=output_file,
        )
        logger = logging.getLogger("speech_recognize")
        if output_file is not sys.stdout:  # also print to stdout
            logger.addHandler(logging.StreamHandler(sys.stdout))
    
    # init environment
    utils.init_env(cfg)

    # build model
    state = checkpoint_utils.load_checkpoint_to_cpu(cfg.common_eval.path)
    saved_cfg = state["cfg"]
    default_config = AudioVideoModelConfig()
    saved_cfg.model = utils.merge_with_parent(default_config, saved_cfg.model)
    model = AudioVideoModel(saved_cfg.model, saved_cfg.task)
    model.load_state_dict(state["model"], strict=True, model_cfg=saved_cfg.model)
    model.eval().cuda()
    model.prepare_for_inference_(cfg)

    saved_cfg.task.modalities = cfg.override.modalities
    logger.info(cfg)

    # Set dictionary
    dictionary = model.tgt_dict

    # build dataset
    if cfg.override.data is not None:
        cfg.task.data = cfg.override.data
    if cfg.override.label_dir is not None:
        cfg.task.label_dir = cfg.override.label_dir
    default_config = TaskConfig()
    saved_cfg.task = utils.merge_with_parent(default_config, saved_cfg.task)
    test_dataset = AudioVideoDataset(saved_cfg.task, cfg.dataset.gen_subset, model.tgt_dict)

    # build iterator
    itr = utils.get_batch_iterator(
        dataset=test_dataset,
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            (sys.maxsize, sys.maxsize), model.max_positions()
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()
    extra_gen_cls_kwargs = {
        "lm_model": None,
        "lm_weight": cfg.generation.lm_weight,
    }
    search_strategy = search.BeamSearch(dictionary)
    generator = SequenceGenerator(
        [model],
        dictionary,
        beam_size=cfg.generation.beam,
        max_len_a=cfg.generation.max_len_a,
        max_len_b=cfg.generation.max_len_b,
        min_len=cfg.generation.min_len,
        normalize_scores=cfg.generation.unnormalized,
        len_penalty=cfg.generation.lenpen,
        unk_penalty=cfg.generation.unkpen,
        temperature=cfg.generation.temperature,
        match_source_len=cfg.generation.match_source_len,
        no_repeat_ngram_size=cfg.generation.no_repeat_ngram_size,
        search_strategy=search_strategy,
        **extra_gen_cls_kwargs,
    )

    def decode_fn(x):
        symbols_ignore = get_symbols_to_strip_from_output(generator)
        symbols_ignore.add(dictionary.pad())

        if hasattr(test_dataset.bpe_tokenizer, 'decode'):
            x = test_dataset.dictionary.string(x, extra_symbols_to_ignore=symbols_ignore)
            return test_dataset.bpe_tokenizer.decode(x)
        chars = dictionary.string(x, extra_symbols_to_ignore=symbols_ignore)
        words = " ".join("".join(chars.split()).replace('|', ' ').split())
        return words

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    result_dict = {'utt_id': [], 'ref': [], 'hypo': []}
    for sample in progress:
        sample = utils.move_to_cuda(sample)
        if "net_input" not in sample:
            continue

        gen_timer.start()
        with torch.no_grad():
            hypos = generator.generate([model], sample)

        num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
        gen_timer.stop(num_generated_tokens)

        for i in range(len(sample["id"])):
            result_dict['utt_id'].append(sample['utt_id'][i])
            ref_sent = decode_fn(sample['target'][i].int().cpu())
            result_dict['ref'].append(ref_sent)
            best_hypo = hypos[i][0]['tokens'].int().cpu()
            hypo_str = decode_fn(best_hypo)
            result_dict['hypo'].append(hypo_str)
            logger.info(f"\nREF:{ref_sent}\nHYP:{hypo_str}\n")
        wps_meter.update(num_generated_tokens)
        progress.log({"wps": round(wps_meter.avg)})
        num_sentences += sample["nsentences"] if "nsentences" in sample else sample["id"].numel()

    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info("Recognized {:,} utterances ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))

    yaml_str = OmegaConf.to_yaml(cfg.generation)
    fid = int(hashlib.md5(yaml_str.encode("utf-8")).hexdigest(), 16)
    fid = fid % 1000000
    result_fn = f"{cfg.common_eval.results_path}/hypo-{fid}.json"
    json.dump(result_dict, open(result_fn, 'w'), indent=4)
    n_err, n_total = 0, 0
    assert len(result_dict['hypo']) == len(result_dict['ref'])
    for hypo, ref in zip(result_dict['hypo'], result_dict['ref']):
        hypo, ref = hypo.strip().split(), ref.strip().split()
        n_err += editdistance.eval(hypo, ref)
        n_total += len(ref)
    wer = 100 * n_err / n_total
    wer_fn = f"{cfg.common_eval.results_path}/wer.{fid}"
    with open(wer_fn, "w") as fo:
        fo.write(f"WER: {wer}\n")
        fo.write(f"err / num_ref_words = {n_err} / {n_total}\n\n")
        fo.write(f"{yaml_str}")
    logger.info(f"WER: {wer}%")
    return


if __name__ == "__main__":
    # init config
    cfg_name = get_args().config_name or "infer"
    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=InferConfig)
    for k in InferConfig.__dataclass_fields__:
        if is_dataclass(InferConfig.__dataclass_fields__[k].type):
            v = InferConfig.__dataclass_fields__[k].default
            cs.store(name=k, node=v)

    main()