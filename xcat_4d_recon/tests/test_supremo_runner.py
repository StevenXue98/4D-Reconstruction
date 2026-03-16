"""Tests for SuPReMo variant configuration and command construction."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from methods.supremo.variants import get_variant_configs, SupremoVariantConfig
from methods.supremo.runner import build_supremo_command, build_animate_command


def _make_cfg(**kwargs) -> SupremoVariantConfig:
    defaults = dict(
        variant_name="test",
        optimizer_type=0,
        surrogate_file="./data/surr.txt",
        output_dir="/tmp/test_out",
        dynamic_image_files="./data/dynamic_image_files.txt",
        ref_image="./data/ref.nii.gz",
        binary="./runSupremo",
        animate_binary="./animate",
        n_threads=2,
    )
    defaults.update(kwargs)
    return SupremoVariantConfig(**defaults)


def test_supremo_cmd_contains_binary():
    cfg = _make_cfg()
    cmd = build_supremo_command(cfg)
    assert cmd[0] == "./runSupremo"


def test_supremo_cmd_optimizer_type():
    for opt_type in [0, 2]:
        cfg = _make_cfg(optimizer_type=opt_type)
        cmd = build_supremo_command(cfg)
        idx = cmd.index("-optimiserType")
        assert cmd[idx + 1] == str(opt_type)


def test_supremo_cmd_surrogate_file():
    cfg = _make_cfg(surrogate_file="/data/my_surr.txt")
    cmd = build_supremo_command(cfg)
    idx = cmd.index("-surr")
    assert cmd[idx + 1] == "/data/my_surr.txt"


def test_animate_cmd_contains_motion_model(tmp_path):
    cfg = _make_cfg(output_dir=str(tmp_path))
    cmd = build_animate_command(cfg, cfg.dynamic_image_files)
    assert "./animate" == cmd[0]
    assert "-motionModel" in cmd


def test_get_variant_configs_keys():
    configs = get_variant_configs(
        data_dir="./data",
        output_dir="./outputs",
        binary="./runSupremo",
        animate_binary="./animate",
    )
    assert set(configs.keys()) == {"surrogate_driven", "surrogate_free", "surrogate_optimized"}


def test_variant_optimizer_types():
    configs = get_variant_configs(
        data_dir="./data",
        output_dir="./outputs",
        binary="./runSupremo",
        animate_binary="./animate",
    )
    assert configs["surrogate_driven"].optimizer_type == 0
    assert configs["surrogate_free"].optimizer_type == 2
    assert configs["surrogate_optimized"].optimizer_type == 2


def test_variant_train_only_file_selection():
    configs = get_variant_configs(
        data_dir="./data",
        output_dir="./outputs",
        binary="./runSupremo",
        animate_binary="./animate",
        train_only=True,
    )
    for cfg in configs.values():
        assert "train" in cfg.dynamic_image_files
