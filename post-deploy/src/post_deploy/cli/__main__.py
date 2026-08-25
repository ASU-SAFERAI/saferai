"""CLI entrypoint for post-deploy."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from post_deploy import __version__
from post_deploy.core.config import PipelineConfig
from post_deploy.core.pipeline import Pipeline
from post_deploy.core.registry import default_registry
from post_deploy.metrics.keyword_search import KeywordSearchMetric
from post_deploy.metrics.pii_search import PiiSearchMetric
from post_deploy.metrics.zero_shot import ZeroShotMetric


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure logging for the CLI."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def register_builtin_metrics() -> None:
    """Register all built-in metrics with the global registry."""
    default_registry.register(KeywordSearchMetric)
    default_registry.register(PiiSearchMetric)
    default_registry.register(ZeroShotMetric)


def build_input_source(config: PipelineConfig):
    """Build an InputSource from the pipeline config."""
    from post_deploy.io.local import LocalCSVInputSource

    input_cfg = config.input

    if input_cfg.type == "local":
        return LocalCSVInputSource(input_cfg.paths)
    elif input_cfg.type == "s3":
        from post_deploy.io.s3 import S3InputSource

        return S3InputSource(
            bucket=input_cfg.bucket or "",
            prefix=input_cfg.prefix or "",
        )
    else:
        raise ValueError(f"Unknown input type: '{input_cfg.type}'")


def build_output_manager(config: PipelineConfig):
    """Build an OutputManager from the pipeline config."""
    from post_deploy.io.local import LocalOutputManager

    output_cfg = config.output

    if output_cfg.type == "local":
        return LocalOutputManager(output_cfg.dir)
    elif output_cfg.type == "s3":
        from post_deploy.io.s3 import S3OutputManager

        return S3OutputManager(
            bucket=output_cfg.bucket or "",
            prefix=output_cfg.prefix or "output",
        )
    else:
        raise ValueError(f"Unknown output type: '{output_cfg.type}'")


def load_preset_config(preset_name: str) -> PipelineConfig:
    """Load a PipelineConfig from a named preset."""
    from post_deploy.presets.safer import PRESET_DIR

    if preset_name == "safer":
        preset_path = PRESET_DIR / "pipeline.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset config not found: {preset_path}")
        return PipelineConfig.from_yaml(preset_path)
    else:
        raise ValueError(f"Unknown preset: '{preset_name}'. Available: ['safer']")


def cmd_run(args: argparse.Namespace) -> None:
    """Execute the 'run' command."""
    register_builtin_metrics()

    # Load config from file or preset
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    elif args.preset:
        config = load_preset_config(args.preset)
    else:
        print("Error: Either --config or --preset is required.", file=sys.stderr)
        sys.exit(1)

    # CLI overrides
    if args.file_paths:
        config.input.paths = [p.strip() for p in args.file_paths.split(",")]

    if args.input_type:
        config.input.type = args.input_type

    if args.output_dir:
        config.output.dir = args.output_dir

    # Build I/O
    input_source = build_input_source(config)
    output_manager = build_output_manager(config)

    # Run pipeline
    logging.info("post-deploy v%s starting.", __version__)
    start = time.time()

    pipeline = Pipeline(
        config=config,
        input_source=input_source,
        output_manager=output_manager,
    )
    pipeline.run()

    elapsed = time.time() - start
    logging.info("Pipeline complete in %.2f seconds.", elapsed)


def cmd_list_metrics(args: argparse.Namespace) -> None:
    """Execute the 'list-metrics' command."""
    register_builtin_metrics()
    metrics = default_registry.list_metrics()

    print(f"Available metrics ({len(metrics)}):")
    for name in metrics:
        metric_cls = default_registry.get(name)
        print(f"  - {name}")


def cmd_validate(args: argparse.Namespace) -> None:
    """Execute the 'validate' command to check a config file."""
    try:
        config = PipelineConfig.from_yaml(args.config)
        print(f"Config is valid: {args.config}")
        print(f"  Input type: {config.input.type}")
        print(f"  Metrics: {[m.name for m in config.get_enabled_metrics()]}")
        print(f"  Output type: {config.output.type}")
        print(f"  Output format: {config.output.format.value}")
    except Exception as e:
        print(f"Config validation failed: {e}", file=sys.stderr)
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="post-deploy",
        description="A pluggable metric framework for evaluating text data.",
    )
    parser.add_argument("--version", action="version", version=f"post-deploy {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run command ---
    run_parser = subparsers.add_parser("run", help="Run the metric pipeline")
    run_parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to pipeline YAML config file.",
    )
    run_parser.add_argument(
        "--preset",
        type=str,
        choices=["safer"],
        help="Use a named preset configuration.",
    )
    run_parser.add_argument(
        "--file-paths",
        type=str,
        help="Comma-separated input file paths (overrides config).",
    )
    run_parser.add_argument(
        "--input-type",
        type=str,
        choices=["local", "s3"],
        help="Input source type (overrides config).",
    )
    run_parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides config).",
    )
    run_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    run_parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file (in addition to stdout).",
    )
    run_parser.set_defaults(func=cmd_run)

    # --- list-metrics command ---
    list_parser = subparsers.add_parser("list-metrics", help="List available metrics")
    list_parser.set_defaults(func=cmd_list_metrics)

    # --- validate command ---
    validate_parser = subparsers.add_parser("validate", help="Validate a config file")
    validate_parser.add_argument("config", type=str, help="Path to the config YAML file.")
    validate_parser.set_defaults(func=cmd_validate)

    return parser


def main() -> None:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Set up logging for the run command
    if args.command == "run":
        setup_logging(
            level=getattr(args, "log_level", "INFO"),
            log_file=getattr(args, "log_file", None),
        )

    # Execute the command
    args.func(args)


if __name__ == "__main__":
    main()
