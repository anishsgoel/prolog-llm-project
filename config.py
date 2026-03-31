"""
Configuration settings for the Prolog LLM reasoning system.
Automatically loaded and logged when the algorithm is run.
"""

import json
import os
import sys

MODEL = "gpt-oss:20b"

DEBUG = False
VERBOSE = True

LLM_TEMPERATURE = 0.0
LLM_NUM_PREDICT = 256
LLM_STOP_TOKENS = ["\n\n", "\nWe ", "\nI ", "\nExplanation", "```"]

DEFAULT_MAX_DEPTH = 10
DEFAULT_MAX_DEPTH_SHORTCUT = 30

HYP_MAX_ATOMS = 1
HYP_MAX_HYP_PER_ATOM = 2
HYP_PROMPT_FACT_LIMIT = 25

SOFT_BFS_MAX_SOLUTIONS = 25
PATH_AGGREGATION_FUNCTION = min


def load_config_from_file(filepath: str = "config.json") -> dict:
    """Load configuration from a JSON file, if it exists."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}


def get_config_log() -> dict:
    """Return a dictionary of all current config values for logging."""
    return {
        "model": MODEL,
        "debug": DEBUG,
        "verbose": VERBOSE,
        "llm_temperature": LLM_TEMPERATURE,
        "llm_num_predict": LLM_NUM_PREDICT,
        "llm_stop_tokens": LLM_STOP_TOKENS,
        "default_max_depth": DEFAULT_MAX_DEPTH,
        "default_max_depth_shortcut": DEFAULT_MAX_DEPTH_SHORTCUT,
        "hyp_max_atoms": HYP_MAX_ATOMS,
        "hyp_max_hyp_per_atom": HYP_MAX_HYP_PER_ATOM,
        "hyp_prompt_fact_limit": HYP_PROMPT_FACT_LIMIT,
        "soft_bfs_max_solutions": SOFT_BFS_MAX_SOLUTIONS,
        "path_aggregation_function": PATH_AGGREGATION_FUNCTION.__name__,
    }


def log_config():
    """Print the current configuration to stdout."""
    config = get_config_log()
    print("=" * 50)
    print("CONFIGURATION")
    print("=" * 50)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 50)


def apply_config_from_file(filepath: str = "config.json"):
    """
    Apply configuration from a JSON file, overriding defaults.
    This allows users to customize settings without modifying code.
    """
    file_config = load_config_from_file(filepath)
    if not file_config:
        return

    global MODEL, DEBUG, VERBOSE
    global LLM_TEMPERATURE, LLM_NUM_PREDICT, LLM_STOP_TOKENS
    global DEFAULT_MAX_DEPTH, DEFAULT_MAX_DEPTH_SHORTCUT
    global HYP_MAX_ATOMS, HYP_MAX_HYP_PER_ATOM, HYP_PROMPT_FACT_LIMIT
    global SOFT_BFS_MAX_SOLUTIONS, PATH_AGGREGATION_FUNCTION

    if "model" in file_config:
        MODEL = file_config["model"]
    if "debug" in file_config:
        DEBUG = file_config["debug"]
    if "verbose" in file_config:
        VERBOSE = file_config["verbose"]
    if "llm_temperature" in file_config:
        LLM_TEMPERATURE = file_config["llm_temperature"]
    if "llm_num_predict" in file_config:
        LLM_NUM_PREDICT = file_config["llm_num_predict"]
    if "llm_stop_tokens" in file_config:
        LLM_STOP_TOKENS = file_config["llm_stop_tokens"]
    if "default_max_depth" in file_config:
        DEFAULT_MAX_DEPTH = file_config["default_max_depth"]
    if "default_max_depth_shortcut" in file_config:
        DEFAULT_MAX_DEPTH_SHORTCUT = file_config["default_max_depth_shortcut"]
    if "hyp_max_atoms" in file_config:
        HYP_MAX_ATOMS = file_config["hyp_max_atoms"]
    if "hyp_max_hyp_per_atom" in file_config:
        HYP_MAX_HYP_PER_ATOM = file_config["hyp_max_hyp_per_atom"]
    if "hyp_prompt_fact_limit" in file_config:
        HYP_PROMPT_FACT_LIMIT = file_config["hyp_prompt_fact_limit"]
    if "soft_bfs_max_solutions" in file_config:
        SOFT_BFS_MAX_SOLUTIONS = file_config["soft_bfs_max_solutions"]
    if "path_aggregation_function" in file_config:
        agg_name = file_config["path_aggregation_function"]
        if agg_name == "min":
            PATH_AGGREGATION_FUNCTION = min
        elif agg_name == "max":
            PATH_AGGREGATION_FUNCTION = max
        else:
            raise ValueError(f"Unsupported path_aggregation_function: {agg_name}")


def init_config(config_file: str = "config.json"):
    """
    Initialize configuration: load from file if exists, then log settings.
    Call this once at program start.
    """
    apply_config_from_file(config_file)
    log_config()
