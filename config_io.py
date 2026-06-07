"""Locate, load and save the project config.yaml (single source of truth)."""
import os
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def config_path():
    return os.path.join(ROOT, "config.yaml")


def load_config():
    p = config_path()
    if not os.path.exists(p):
        raise FileNotFoundError(
            "config.yaml not found. Run `python config.py` first.")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg):
    with open(config_path(), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
