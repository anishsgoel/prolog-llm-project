"""
Configuration settings for the Prolog LLM reasoning system.
Module-level variables are derived from GlobalConfig and can still be
imported as ``import config; config.VERBOSE`` throughout the codebase.
Use ``apply_global_config()`` or ``init_config()`` to load overrides.
"""

from cfg import GlobalConfig

_cfg = GlobalConfig()

MODEL = _cfg.llm.model
DEBUG = _cfg.debug
VERBOSE = _cfg.verbose

LLM_TEMPERATURE = _cfg.llm.temperature
LLM_NUM_PREDICT = _cfg.llm.num_predict
LLM_STOP_TOKENS = _cfg.llm.stop_tokens

DEFAULT_MAX_DEPTH = _cfg.default_max_depth
DEFAULT_MAX_DEPTH_SHORTCUT = _cfg.default_max_depth_shortcut

HYP_MAX_ATOMS = _cfg.hyp_max_atoms
HYP_MAX_HYP_PER_ATOM = _cfg.hyp_max_hyp_per_atom
HYP_PROMPT_FACT_LIMIT = _cfg.hyp_prompt_fact_limit

SOFT_BFS_MAX_SOLUTIONS = _cfg.soft_bfs_max_solutions
PATH_AGGREGATION_FUNCTION = min if _cfg.path_aggregation == "min" else max


def apply_global_config(cfg: GlobalConfig) -> None:
    """Push a GlobalConfig into the module-level variables."""
    global MODEL, DEBUG, VERBOSE
    global LLM_TEMPERATURE, LLM_NUM_PREDICT, LLM_STOP_TOKENS
    global DEFAULT_MAX_DEPTH, DEFAULT_MAX_DEPTH_SHORTCUT
    global HYP_MAX_ATOMS, HYP_MAX_HYP_PER_ATOM, HYP_PROMPT_FACT_LIMIT
    global SOFT_BFS_MAX_SOLUTIONS, PATH_AGGREGATION_FUNCTION

    MODEL = cfg.llm.model
    DEBUG = cfg.debug
    VERBOSE = cfg.verbose
    LLM_TEMPERATURE = cfg.llm.temperature
    LLM_NUM_PREDICT = cfg.llm.num_predict
    LLM_STOP_TOKENS = cfg.llm.stop_tokens
    DEFAULT_MAX_DEPTH = cfg.default_max_depth
    DEFAULT_MAX_DEPTH_SHORTCUT = cfg.default_max_depth_shortcut
    HYP_MAX_ATOMS = cfg.hyp_max_atoms
    HYP_MAX_HYP_PER_ATOM = cfg.hyp_max_hyp_per_atom
    HYP_PROMPT_FACT_LIMIT = cfg.hyp_prompt_fact_limit
    SOFT_BFS_MAX_SOLUTIONS = cfg.soft_bfs_max_solutions
    PATH_AGGREGATION_FUNCTION = min if cfg.path_aggregation == "min" else max


def init_config(config_file: str = "global.yaml") -> GlobalConfig:
    """Load a GlobalConfig from YAML (if it exists) and apply it."""
    try:
        cfg = GlobalConfig.from_yaml(config_file)
    except FileNotFoundError:
        cfg = GlobalConfig()
    apply_global_config(cfg)
    _log(cfg)
    return cfg


def _log(cfg: GlobalConfig) -> None:
    print("=" * 50)
    print("CONFIGURATION")
    print("=" * 50)
    for key, value in cfg.model_dump().items():
        print(f"  {key}: {value}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Legacy helpers kept for backward compatibility
# ---------------------------------------------------------------------------

def get_config_log() -> dict:
    return GlobalConfig(
        llm={"model": MODEL, "temperature": LLM_TEMPERATURE,
             "num_predict": LLM_NUM_PREDICT, "stop_tokens": LLM_STOP_TOKENS},
        debug=DEBUG, verbose=VERBOSE,
        hyp_prompt_fact_limit=HYP_PROMPT_FACT_LIMIT,
        soft_bfs_max_solutions=SOFT_BFS_MAX_SOLUTIONS,
        path_aggregation="min" if PATH_AGGREGATION_FUNCTION is min else "max",
        default_max_depth=DEFAULT_MAX_DEPTH,
        default_max_depth_shortcut=DEFAULT_MAX_DEPTH_SHORTCUT,
        hyp_max_atoms=HYP_MAX_ATOMS,
        hyp_max_hyp_per_atom=HYP_MAX_HYP_PER_ATOM,
    ).model_dump()


def log_config() -> None:
    for key, value in get_config_log().items():
        print(f"  {key}: {value}")