"""
Configuration settings for the Prolog LLM reasoning system.
Module-level variables are derived from GlobalConfig and can still be
imported as ``import config; config.VERBOSE`` throughout the codebase.
Use ``apply_global_config()`` or ``init_config()`` to load overrides.
"""

from cfg import GlobalConfig
from log_setup import setup_logging

_cfg = GlobalConfig()

MODEL = _cfg.llm.model
DEBUG = _cfg.debug
VERBOSE = _cfg.verbose

LLM_TEMPERATURE = _cfg.llm.temperature
LLM_NUM_PREDICT = _cfg.llm.num_predict
LLM_STOP_TOKENS = _cfg.llm.stop_tokens

DEFAULT_MAX_DEPTH = _cfg.default_max_depth

HYP_PROMPT_FACT_LIMIT = _cfg.hyp_prompt_fact_limit

PATH_AGGREGATION_FUNCTION = min if _cfg.path_aggregation == "min" else max

_logged = False


def apply_global_config(cfg: GlobalConfig) -> None:
    """Push a GlobalConfig into the module-level variables."""
    global MODEL, DEBUG, VERBOSE
    global LLM_TEMPERATURE, LLM_NUM_PREDICT, LLM_STOP_TOKENS
    global DEFAULT_MAX_DEPTH, HYP_PROMPT_FACT_LIMIT, PATH_AGGREGATION_FUNCTION

    MODEL = cfg.llm.model
    DEBUG = cfg.debug
    VERBOSE = cfg.verbose
    LLM_TEMPERATURE = cfg.llm.temperature
    LLM_NUM_PREDICT = cfg.llm.num_predict
    LLM_STOP_TOKENS = cfg.llm.stop_tokens
    DEFAULT_MAX_DEPTH = cfg.default_max_depth
    HYP_PROMPT_FACT_LIMIT = cfg.hyp_prompt_fact_limit
    PATH_AGGREGATION_FUNCTION = min if cfg.path_aggregation == "min" else max


def init_config(config_file: str = "global.yaml") -> GlobalConfig:
    """Load a GlobalConfig from YAML (if it exists), apply it, and set up logging."""
    global _logged

    try:
        cfg = GlobalConfig.from_yaml(config_file)
    except FileNotFoundError:
        cfg = GlobalConfig()
    apply_global_config(cfg)
    setup_logging(verbose=cfg.verbose, debug=cfg.debug)

    # Print the resolved configuration only once per process, even across
    # repeated init_config() calls (e.g. one per problem in a batch run).
    if not _logged:
        _log(cfg)
        _logged = True
    return cfg


def _log(cfg: GlobalConfig) -> None:
    print("=" * 50)
    print("CONFIGURATION")
    print("=" * 50)
    for key, value in cfg.model_dump().items():
        print(f"  {key}: {value}")
    print("=" * 50)