from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlcq_graphs.config import dump_resolved_config, load_config, parse_cli_overrides
from mlcq_graphs.pipeline import PipelineRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MLCQ pipeline from YAML configuration.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/gcn_baseline.yml"),
        help="Path to experiment configuration file.",
    )
    parser.add_argument(
        "--resolved-config-out",
        type=Path,
        default=None,
        help="Optional path to write resolved config YAML.",
    )
    parser.add_argument(
        "--print-resolved-config",
        action="store_true",
        help="Print resolved config to stdout before running.",
    )

    args, unknown = parser.parse_known_args()
    overrides = parse_cli_overrides(unknown)
    config = load_config(args.config, overrides)

    project_root = Path(__file__).resolve().parent
    if args.resolved_config_out is not None:
        resolved_out = args.resolved_config_out
    else:
        artifacts_root = Path(str(config.get("run", {}).get("artifacts_root", "artifacts")))
        resolved_out = project_root / artifacts_root / "resolved_config.yml"

    dump_resolved_config(config, resolved_out)

    if args.print_resolved_config:
        print(json.dumps(config, indent=2))

    result = PipelineRunner(config=config, project_root=project_root).run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
