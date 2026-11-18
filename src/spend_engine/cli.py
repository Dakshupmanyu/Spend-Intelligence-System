from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import ConfigurationError, load_config
from .pipeline import PipelineError, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spend-engine", description="Config-driven spend intelligence engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run one company end-to-end")
    run.add_argument("--config", required=True, help="Company YAML configuration")
    run.add_argument("--output-dir", default="outputs", help="Directory for final CSV and JSON report")
    run.add_argument("--candidate-name", default="candidate", help="Prefix used in output filenames")
    run.add_argument("--output-tag", default=None, help="Optional output suffix, e.g. A2 or B2, without changing the company config")
    run.add_argument("--disable-label-memory", action="store_true", help="Run a strict no-label classification test")
    run.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config, config_dir = load_config(args.config)
        output_dir = Path(args.output_dir).expanduser().resolve()
        report = run_pipeline(
            config=config,
            config_dir=config_dir,
            output_dir=output_dir,
            candidate_name=args.candidate_name,
            disable_label_memory=args.disable_label_memory,
            output_tag=args.output_tag,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except (ConfigurationError, PipelineError, FileNotFoundError, ValueError, RuntimeError) as exc:
        logging.error("Run failed: %s", exc)
        return 2
    except Exception:
        logging.exception("Unexpected pipeline failure")
        return 3


if __name__ == "__main__":
    sys.exit(main())
