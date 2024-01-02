# Some codes are borrowed from fairseq so that
# we can simplify the project.
import argparse
from argparse import Namespace
from dataclasses import is_dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf, open_dict

import utils
from .arguments import FairseqConfig


def hydra_init(cfg_name):
    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=FairseqConfig)
    for k in FairseqConfig.__dataclass_fields__:
        v = FairseqConfig.__dataclass_fields__[k].default
        try:
            cs.store(name=k, node=v)
        except BaseException:
            logger.error(f"{k} - {v}")
            raise


def merge_with_parent(dc, cfg, remove_missing=True):
    if remove_missing:
        if is_dataclass(dc):
            target_keys = set(dc.__dataclass_fields__.keys())
        else:
            target_keys = set(dc.keys())

        with open_dict(cfg):
            for k in list(cfg.keys()):
                if k not in target_keys:
                    del cfg[k]

    merged_cfg = OmegaConf.merge(dc, cfg)
    merged_cfg.__dict__["_parent"] = cfg.__dict__["_parent"]
    OmegaConf.set_struct(merged_cfg, True)
    return merged_cfg


def flatten_config(cfg):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def overwrite_args_by_name(cfg, overrides):
    # this will be deprecated when we get rid of argparse and model_overrides logic

    from fairseq.registry import REGISTRIES

    with open_dict(cfg):
        for k in cfg.keys():
            # "k in cfg" will return false if its a "mandatory value (e.g. ???)"
            if k in cfg and isinstance(cfg[k], DictConfig):
                if k in overrides and isinstance(overrides[k], dict):
                    for ok, ov in overrides[k].items():
                        if isinstance(ov, dict) and cfg[k][ok] is not None:
                            overwrite_args_by_name(cfg[k][ok], ov)
                        else:
                            cfg[k][ok] = ov
                else:
                    overwrite_args_by_name(cfg[k], overrides)
            elif k in cfg and isinstance(cfg[k], Namespace):
                for override_key, val in overrides.items():
                    setattr(cfg[k], override_key, val)
            elif k in overrides:
                if (
                    k in REGISTRIES
                    and overrides[k] in REGISTRIES[k]["dataclass_registry"]
                ):
                    cfg[k] = DictConfig(
                        REGISTRIES[k]["dataclass_registry"][overrides[k]]
                    )
                    overwrite_args_by_name(cfg[k], overrides)
                    cfg[k]._name = overrides[k]
                else:
                    cfg[k] = overrides[k]