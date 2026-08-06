# Baselines for LLM-guided abductive Prolog KB completion

This document describes the baseline suite that competes head-to-head with the
system under study (**PLLM**, the LLM-guided DFS meta-solver). Every method emits
the **same** `RunMetrics` (`src/experiment/tracking.py`) and plugs into the same
batch runner, so the whole comparison table is directly comparable.

## How the pieces fit together

- `src/experiment/tracking.py`
  - `build_problem(config_name, model)` — loads the config, builds `kb_full` /
    `kb_missing`, parses the goal, and runs the time-bounded ground truth. Shared
    by every method so they all start from an identical setup.
  - `compute_metrics(...)` — computes the full `RunMetrics` from a
    DFSMetaSolver-shaped `result` dict. Every method routes through this, so all
    columns are computed identically.
  - `run_tracked(...)` — PLLM (unchanged output; now a thin wrapper over
    `build_problem` + solve + `compute_metrics`).
- `src/experiment/methods.py` — one `run_<name>` per method + the `METHODS` /
  `ABLATIONS` registries and the `run_method(...)` dispatcher.
- `src/experiment/baseline_policies.py` — the constant/candidate-atom
  enumeration helpers, the `RandomFactProposerPolicy`, and the PLLM ablation
  policies.
- Prompt templates: `src/experiment/templates/{oneshot_completion,qa,logic_lm_refine}_prompt.j2`.

### Metrics to read (and to ignore)

Report **`valid_success`** (proof found AND every leaf is a real KB fact),
**`proof_matches_gt`** (proof leaf-set equals the full-KB solver's; `"NA"` when
GT timed out), and **`precision_hypothesized` / `recall_omitted` /
`hallucination_rate`** over the proposed facts. Do **not** headline plain
`success`: the solver can nearly always inject a fact to force *a* proof, so
`success` is ~always 1 and uninformative.

Two implementation notes that keep the table comparable:

- **Verification prover.** The baselines verify provability with `DFSSolver`
  (the exact prover PLLM uses) under a no-op policy with `allow_extension=False`,
  not the priority-queue `Solver`. The priority-queue solver blows up
  combinatorially at the ceiling depth on the recursive rules (tens of seconds
  per failing probe), which `minimal_alp` would call thousands of times. The
  ground truth still uses the classic `Solver` in a time-bounded subprocess, as
  before.
- **Canonical atom comparison.** `compute_metrics` compares proposed facts to the
  true/omitted facts through `AtomicFormula.__str__` on both sides (matching the
  soundness block), so precision/recall are robust to spacing. This is
  numerically identical to the old raw-string comparison for every KB in the repo
  (all write facts with a single `", "`), so PLLM numbers are unchanged.

## The methods

Run with `--method <name>` (default `pllm`). Ablations run with `--ablation <name>`
(a PLLM variant; overrides `--method`).

### `pllm`
The system under study: LLM-guided DFS meta-solver that hypothesises missing
facts (with confidences) to recover a proof. Everything else is a comparison
point against it.

### `no_abduction` (floor)
Plain deduction on the ablated KB with **no** fact injection. Isolates "is the
problem actually unsolvable without the omitted facts?". **Expected:** `success`
== 0 on every config; a success here means the omitted facts are not
load-bearing (the runner prints a warning).

### `random_edge`
The DFS meta-solver, identical to PLLM, but the "LLM" proposes **random** candidate
facts (seeded by `run_index`). Isolates the search machinery from the LLM's world
knowledge: does the LLM's knowledge matter, or just the scaffold? **Expected:** it
often "succeeds" via a lucky shortcut edge but scores ~0 `recall_omitted` and high
`hallucination_rate` — that contrast is the point.

### `minimal_alp` (symbolic abduction)
Classical minimum-cardinality abduction, **no LLM**: over the candidate
ground-atom pool it searches increasing |Δ| for the smallest fact set that makes
the goal provable. Isolates what pure symbolic search recovers with no world
knowledge. **Expected:** high `success` with minimal proofs, but low
precision/recall — it closes the proof with whatever works, typically a shortcut.
`K_MAX` defaults to `|omit_fact_ids| + 1` (hard-capped at 3); the runner logs when
capped. The candidate pool is restricted to constants already in the KB (there is
no vocabulary of "new" constants to enumerate), so on `allow_new_constants=true`
problems it can only rediscover facts over the existing constants.

Cites: Kakas, Kowalski & Toni, *Abductive Logic Programming*, J. Logic and
Computation 2(6), 1992; Denecker & Kakas, *Abduction in Logic Programming*, LNCS
2407, Springer, 2002.

### `llm_oneshot`
One LLM call proposes the missing facts; the **symbolic solver** then verifies
they yield a proof. Isolates whether the iterative per-node DFS scaffold beats a
single prompt.

### `llm_qa`
One LLM call answers the goal true/false — **no solver, no proof**. Isolates
whether a solver is needed at all. Its figure of merit is `success` (the predicted
boolean) vs `ground_truth_success`. Because it produces no proof,
`valid_success` / `proof_matches_gt` / `precision` / `recall` / `hallucination`
are all 0 here — that is **N/A for QA, not a failure**.

### `logic_lm` (literature baseline)
Logic-LM-style: the LLM formalises candidate facts, the symbolic solver verifies,
and on failure a self-refinement round re-proposes from the still-open query, up
to `refine_budget` (default 3) rounds. `refine_budget=0` collapses to
`llm_oneshot`, giving a clean within-baseline gradient. `llm_queries_formalize`
and `llm_queries_refine` are counted separately. This is the strong "LLM + solver,
monolithic + self-refine" comparison against PLLM's decomposed per-node search.

Cites: Pan, Albalak, Wang & Wang, *Logic-LM: Empowering Large Language Models with
Symbolic Solvers for Faithful Logical Reasoning*, Findings of EMNLP 2023
(arXiv:2305.12295). Labelled "Logic-LM-style" because here the KB is already
symbolic, so only the propose→solve→refine loop is adapted (not the NL→logic
formalisation).

## PLLM ablations

Run with `--ablation <name>`:

- `verify_on` — turn on the LLM keep/drop self-critique of each hypothesis
  (`config.VERIFY_HYPOTHESES=True`, set before the policy is built because it is
  snapshotted at construction).
- `min_hypotheses` — switch the proof objective to fewest-hypotheses
  (`config.PROOF_OBJECTIVE="min_hypotheses"`).
- `no_order` — disable the LLM goal-ordering step (keep DFS input order).
- `fixed_depth` — replace the LLM depth estimate with `DEFAULT_MAX_DEPTH`.
- `init_only` — hypothesise only once, before search (never on backtrack).
- `backtrack_only` — hypothesise only during backtracking (not before search).
- `self_consistency` *(optional)* — sample `k` proposal sets (default 3) and union
  them before a single symbolic verification. **Caveat:** `LLMInterface.generate`
  currently comments out the Ollama `options` block, so an explicit `temperature>0`
  cannot be forced; this relies on the backend's default sampler nondeterminism to
  make the `k` proposals differ.

## Reproducing the comparison table

Each command runs on all configs (`--configs "*"`), each config with each model
(`--models`), writing a CSV (and optional `.xlsx`) plus MLflow. Drop `--no-mlflow`
to log to MLflow; add `--xlsx results.xlsx` for an Excel workbook with a `summary`
sheet (means per config × model × method × ablation).

```bash
# PLLM (the system under study), across models
python src/experiment/batch_run.py --method pllm \
    --models gpt-oss:20b --output results_pllm.csv

# Standalone baselines
python src/experiment/batch_run.py --method no_abduction --output results_no_abduction.csv
python src/experiment/batch_run.py --method minimal_alp  --output results_minimal_alp.csv
python src/experiment/batch_run.py --method random_edge  --runs 5 --output results_random_edge.csv
python src/experiment/batch_run.py --method llm_oneshot  --models gpt-oss:20b --output results_llm_oneshot.csv
python src/experiment/batch_run.py --method llm_qa       --models gpt-oss:20b --output results_llm_qa.csv
python src/experiment/batch_run.py --method logic_lm     --models gpt-oss:20b --output results_logic_lm.csv

# PLLM ablations
for A in verify_on min_hypotheses no_order fixed_depth init_only backtrack_only self_consistency; do
    python src/experiment/batch_run.py --ablation "$A" --models gpt-oss:20b --output "results_ablation_$A.csv"
done

# A quick, LLM-free smoke on two small, fast configs
python src/experiment/batch_run.py --method minimal_alp \
    --configs london_directed bio_grn_p53 --no-mlflow --output smoke_minimal_alp.csv
```

Notes:
- `random_edge` uses `run_index` as its RNG seed, so `--runs N` gives N distinct,
  reproducible random baselines. It needs no LLM.
- `no_abduction`, `minimal_alp`, `random_edge` need **no** Ollama model; the
  `llm_*` methods, `logic_lm`, and all ablations do.
- Each evaluation runs in a killable subprocess under `--eval-timeout` (default
  7200s). The ground-truth search has its own 300s budget; on timeout the
  GT-derived columns are recorded as `NA`.