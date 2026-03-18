from __future__ import annotations

import argparse
from typing import Sequence

from . import config as workflow_config
from .pipeline import PipelineResult, run_pipeline


def build_cli_arg_parser() -> argparse.ArgumentParser:
    parser = workflow_config.build_workflow_arg_parser()
    parser.prog = "scbiomarker"
    parser.description = "Run the scbiomarker workflow from the command line."
    parser.add_argument("adata_input", nargs="?", default=None, help="Input .h5ad path.")
    parser.add_argument(
        "--pathway-path",
        "--pathway",
        dest="pathway_path",
        default=None,
        help="Pathway gene-set file for biomarker analysis.",
    )
    parser.add_argument(
        "--k",
        dest="k",
        type=int,
        default=None,
        help="Number of preselected genes to keep for training/analysis.",
    )
    parser.add_argument(
        "--gpu",
        dest="gpu_index",
        type=int,
        default=None,
        help="GPU index for runtime selection. Use -1 to force CPU analysis.",
    )
    return parser


def _apply_positional_adata_argument(args: argparse.Namespace) -> None:
    positional_value = getattr(args, "adata_input", None)
    if positional_value in {None, ""}:
        return

    option_value = getattr(args, "input_h5ad", None)
    if option_value not in {None, ""}:
        raise ValueError("Provide either positional <input_h5ad> or --adata, not both.")
    args.input_h5ad = positional_value


def _result_lines(result: PipelineResult) -> list[str]:
    lines = [
        f"phases={','.join(result.phases_completed)}",
        f"input_h5ad={result.input_h5ad}",
        f"splits_directory={result.splits_directory}",
        f"preselection_output_root={result.preselection_output_root}",
        f"run_dir={result.run_dir}",
        f"analysis_output_dir={result.analysis_output_dir}",
        f"config_snapshot_path={result.config_snapshot_path}",
    ]
    if result.training_artifacts is not None and "best_checkpoint_path" in result.training_artifacts:
        lines.append(f"best_checkpoint_path={result.training_artifacts['best_checkpoint_path']}")
    return lines


def main(argv: Sequence[str] | None = None) -> PipelineResult:
    parser = build_cli_arg_parser()
    args = parser.parse_args(argv)
    _apply_positional_adata_argument(args)

    config_overrides = workflow_config.workflow_overrides_from_args(args)
    if args.k is not None:
        config_overrides.setdefault("training", {})["k"] = int(args.k)

    resolved_config = workflow_config.load_workflow_config_dict(
        config_path=args.config_path,
        overrides=config_overrides if config_overrides else None,
    )
    gpu_index = args.gpu_index
    if gpu_index is None and not bool(resolved_config.get("analysis_only", False)):
        gpu_index = 0

    result = run_pipeline(
        config_path=args.config_path,
        config_overrides=config_overrides,
        pathway_path=args.pathway_path,
        gpu_index=gpu_index,
    )

    for line in _result_lines(result):
        print(line, flush=True)
    return result


__all__ = [
    "build_cli_arg_parser",
    "main",
]
