# AGENTS.md

This is the repository implementing the survey
[A survey on code smells detection using machine learning techniques](https://doi.org/10.1016/j.infsof.2026.108242)
(Information and Software Technology, 2026). It trains and evaluates classical ML,
sequence DL and GNN detectors on MLCQ with the same preprocessing, labels and
evaluation protocol.

People usually clone it for one of two things: reproduce the results of the
survey, or test their own detector under the same protocol. Help them do that.
Start by reading `README.md`.

## Do not touch the actual implementation and interfaces

Results stay comparable with the survey only if the code that produces them is
unchanged. Do not modify, rename or refactor:

- `main.py` and its command line
- `mlcq_graphs/` (pipeline, models, training, evaluation)
- the scripts at the root of `scripts/`, which the pipeline calls
- `config/defaults.yml` and the existing experiment configs
- `data/`, `tools/` and `prompts/`

Add new code next to it instead (see below). If the user asks for a change in
these files, tell them that their results will no longer be comparable with the
survey, and go ahead only if they confirm.

## Architecture

`main.py` reads a YAML config and runs `mlcq_graphs/pipeline.py`. Each stage is
cached with a SHA-256 fingerprint of its inputs, so only the stages whose inputs
changed run again.

1. Normalization. `scripts/NormalizeFromJson.py` turns the MLCQ reviews into one
   binary label per smell.
2. Representation. OO metrics with DesigniteJava, token sequences or AST graphs
   (ANTLR, then DOT, then PyTorch Geometric).
3. Training. `training.used_method` selects classical, sequence, gnn or codebert.
4. Evaluation. Per-label threshold tuning, then F1, MCC and PR-AUC over several
   seeds.

Outputs go to `artifacts/`. Studies outside the main pipeline live in
`scripts/<topic>/`: cv, data, labels, llm, analysis, pretraining, xai and slurm.
Each of these scripts starts with a docstring that gives the command to run it.

## Common tasks

- Setup: `uv sync`. Python 3.13+, Java 17+ for DesigniteJava. A GPU is
  recommended for the sequence and GNN models.
- Reproduce a model family: use the commands of `README.md`, section
  "Reproducing the paper results".
- Change a hyperparameter: override it on the command line, for example
  `'--training.seeds=[42,43]'`. Do not edit `config/defaults.yml`.
- New experiment: add a YAML in `config/experiments/` that starts with
  `_include_yml: ../defaults.yml`.
- New graph model: add a file in `mlcq_graphs/models/` with a subclass of
  `BaseGraphClassifier` decorated with `@register_model("name")`, import it in
  `mlcq_graphs/models/__init__.py` (the only edit allowed there), then select
  it with `'--training.architecture=name'`.
- Compare several detectors on the same samples: use the shared
  cross-validation protocol (`config/experiments/gnn_cv.yml`,
  `config/experiments/tabular_cv.yml`, then `scripts/cv/summarize_cv.py`).
- Tables and figures:
  `uv run python scripts/report_runs.py --artifacts-root artifacts`.

## Guidelines

- Use `uv run` for every command.
- Keep the label order `[feature_envy, long_method, blob, data_class]`.
- Warn the user before long GPU runs.
- Never present a number that no run produced. To compare with the survey,
  quote the tables of the survey.
- Run the tests after any change:
  `uv run python -m unittest discover -s tests`.
