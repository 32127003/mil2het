from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import scanpy as sc
from scipy import sparse


# -------------------------
# Diagnostics
# -------------------------

@dataclass(frozen=True)
class ExpressionSummary:
    is_sparse: bool
    dtype: str
    minimum: float
    maximum: float
    nonzero_mean: float
    nonzero_median: float
    fraction_zeros: float
    cell_total_mean: float
    cell_total_median: float
    cell_total_std: float
    cell_total_cv: float  # std / mean


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _compute_cell_totals(expression_matrix) -> np.ndarray:
    if sparse.issparse(expression_matrix):
        totals = np.asarray(expression_matrix.sum(axis=1)).ravel()
    else:
        totals = np.asarray(expression_matrix).sum(axis=1)
    return totals.astype(np.float64, copy=False)


def summarize_expression_matrix(expression_matrix) -> ExpressionSummary:
    is_sparse_matrix = sparse.issparse(expression_matrix)
    dtype_name = str(expression_matrix.dtype)

    if is_sparse_matrix:
        data_array = expression_matrix.data
        number_of_entries = int(expression_matrix.shape[0] * expression_matrix.shape[1])
        number_of_nonzeros = int(data_array.size)

        if number_of_nonzeros == 0:
            minimum_value = 0.0
            maximum_value = 0.0
            nonzero_mean = 0.0
            nonzero_median = 0.0
        else:
            minimum_value = _safe_float(data_array.min())
            maximum_value = _safe_float(data_array.max())
            nonzero_mean = _safe_float(data_array.mean())
            nonzero_median = _safe_float(np.median(data_array))

        fraction_zeros = 1.0 - (number_of_nonzeros / max(1, number_of_entries))
    else:
        dense_array = np.asarray(expression_matrix)
        minimum_value = _safe_float(dense_array.min())
        maximum_value = _safe_float(dense_array.max())
        nonzero_values = dense_array[dense_array != 0]
        if nonzero_values.size == 0:
            nonzero_mean = 0.0
            nonzero_median = 0.0
        else:
            nonzero_mean = _safe_float(nonzero_values.mean())
            nonzero_median = _safe_float(np.median(nonzero_values))
        fraction_zeros = _safe_float(np.mean(dense_array == 0))

    cell_totals = _compute_cell_totals(expression_matrix)
    cell_total_mean = _safe_float(cell_totals.mean()) if cell_totals.size else 0.0
    cell_total_median = _safe_float(np.median(cell_totals)) if cell_totals.size else 0.0
    cell_total_std = _safe_float(cell_totals.std(ddof=0)) if cell_totals.size else 0.0
    cell_total_cv = _safe_float(cell_total_std / cell_total_mean) if cell_total_mean > 0 else float("nan")

    return ExpressionSummary(
        is_sparse=is_sparse_matrix,
        dtype=dtype_name,
        minimum=minimum_value,
        maximum=maximum_value,
        nonzero_mean=nonzero_mean,
        nonzero_median=nonzero_median,
        fraction_zeros=fraction_zeros,
        cell_total_mean=cell_total_mean,
        cell_total_median=cell_total_median,
        cell_total_std=cell_total_std,
        cell_total_cv=cell_total_cv,
    )


def infer_expression_state(
    summary: ExpressionSummary,
    has_log1p_metadata: bool,
    target_sum: float,
) -> Dict[str, Any]:
    """
    Heuristic inference:
    - If negative values exist -> "scaled_or_residual" (not valid for log1p pipeline)
    - Else if metadata says log1p -> "log1p"
    - Else if max is small (<= ~25) and dtype is float -> likely log1p
    - Else -> not log1p (counts or normalized but not logged)
    Additionally guess whether per-cell totals look normalized to target_sum.
    """
    if summary.minimum < 0:
        return {
            "state": "scaled_or_residual",
            "looks_log1p": False,
            "looks_normalized_total": False,
            "reason": f"minimum < 0 ({summary.minimum:.3g})",
        }

    # log1p metadata (Scanpy sets adata.uns['log1p'] when sc.pp.log1p is called)
    if has_log1p_metadata:
        return {
            "state": "log1p",
            "looks_log1p": True,
            "looks_normalized_total": False,  # log-space totals aren't meaningful
            "reason": "adata.uns['log1p'] exists",
        }

    # Value-range heuristic for log1p
    dtype_is_integer = ("int" in summary.dtype)
    if (not dtype_is_integer) and (summary.maximum <= 25.0):
        return {
            "state": "log1p_probable",
            "looks_log1p": True,
            "looks_normalized_total": False,
            "reason": f"float dtype + max <= 25 (max={summary.maximum:.3g})",
        }

    # Non-log data: check if totals look normalized
    total_mean = summary.cell_total_mean
    total_cv = summary.cell_total_cv
    looks_normalized_total = (
        np.isfinite(total_mean)
        and np.isfinite(total_cv)
        and (0.8 * target_sum <= total_mean <= 1.2 * target_sum)
        and (total_cv <= 0.2)
    )

    if dtype_is_integer or summary.maximum > 50.0:
        state = "counts_or_large_scale"
    else:
        state = "nonlog_float"

    return {
        "state": state,
        "looks_log1p": False,
        "looks_normalized_total": bool(looks_normalized_total),
        "reason": f"dtype={summary.dtype}, max={summary.maximum:.3g}, cell_total_mean={total_mean:.3g}, cv={total_cv:.3g}",
    }


# -------------------------
# Conversion
# -------------------------

def ensure_log1p_h5ad_inplace(
    input_h5ad_path: str,
    output_h5ad_path: Optional[str] = None,
    preferred_counts_layer: str = "counts",
    target_sum: float = 1e4,
    make_backup: bool = True,
    backup_suffix: str = ".bak",
    force_normalize_total: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Load h5ad, diagnose X (or counts layer), ensure log1p matrix in adata.X, then save.

    Behavior
    - If .layers['counts'] exists: use it as the source for preprocessing (counts → normalize_total → log1p).
    - Else use .X as source.
    - If already log1p (metadata or heuristic): no-op.
    - If non-log and totals don't look normalized: normalize_total(target_sum) then log1p.
      If totals already look normalized and force_normalize_total=False: skip normalize_total, only log1p.
    - Saves to output_h5ad_path. If None, overwrite input path.
    """
    if output_h5ad_path is None:
        output_h5ad_path = input_h5ad_path

    if not os.path.exists(input_h5ad_path):
        raise FileNotFoundError(f"h5ad not found: {input_h5ad_path}")

    adata = sc.read_h5ad(input_h5ad_path)

    # choose source matrix for diagnosis + potential preprocessing
    using_counts_layer = preferred_counts_layer in getattr(adata, "layers", {})
    source_matrix = adata.layers[preferred_counts_layer] if using_counts_layer else adata.X

    source_summary = summarize_expression_matrix(source_matrix)
    has_log1p_metadata = isinstance(adata.uns.get("log1p", None), dict) or ("log1p" in adata.uns)
    inferred = infer_expression_state(source_summary, has_log1p_metadata=has_log1p_metadata, target_sum=float(target_sum))

    report: Dict[str, Any] = {
        "input_path": input_h5ad_path,
        "output_path": output_h5ad_path,
        "used_counts_layer_as_source": bool(using_counts_layer),
        "source_summary": source_summary.__dict__,
        "inference": inferred,
    }

    # If scaled/residual: stop early (this should not be log1p-transformed blindly)
    if inferred["state"] == "scaled_or_residual":
        raise ValueError(
            "Expression matrix has negative values; this looks like scaled/residualized data. "
            "Provide raw counts in adata.layers['counts'] or adata.raw, or set adata.X to non-negative counts."
        )

    # Already log1p -> save (optionally just normalize metadata), but generally no-op
    if inferred["looks_log1p"]:
        if make_backup and (output_h5ad_path == input_h5ad_path):
            backup_path = input_h5ad_path + backup_suffix
            if not os.path.exists(backup_path):
                shutil.copy2(input_h5ad_path, backup_path)
                report["backup_path"] = backup_path

        adata.write_h5ad(output_h5ad_path)
        report["action"] = "no_op_already_log1p"
        return output_h5ad_path, report

    # Need log1p: create a working view where X is source_matrix
    # (If using counts layer, we keep counts in layer and write log1p into X)
    adata.X = source_matrix

    # Ensure float for preprocessing stability
    if sparse.issparse(adata.X):
        adata.X = adata.X.astype(np.float64)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float64)

    # Normalize_total if needed (or forced)
    if force_normalize_total or (not inferred["looks_normalized_total"]):
        sc.pp.normalize_total(adata, target_sum=float(target_sum))
        report["did_normalize_total"] = True
    else:
        report["did_normalize_total"] = False

    sc.pp.log1p(adata)
    report["did_log1p"] = True

    # backup then save
    if make_backup and (output_h5ad_path == input_h5ad_path):
        backup_path = input_h5ad_path + backup_suffix
        if not os.path.exists(backup_path):
            shutil.copy2(input_h5ad_path, backup_path)
            report["backup_path"] = backup_path

    adata.write_h5ad(output_h5ad_path)
    report["action"] = "converted_to_log1p"
    return output_h5ad_path, report


# -------------------------
# Example usage
# -------------------------

if __name__ == "__main__":
    aorta_path = "./aorta/sample_aorta_data_mapped.h5ad"
    asthma_path = "./asthma/asthma_mapped.h5ad"

    for h5ad_path in [aorta_path, asthma_path]:
        saved_path, report = ensure_log1p_h5ad_inplace(
            input_h5ad_path=h5ad_path,
            output_h5ad_path=None,        # None => overwrite same path
            preferred_counts_layer="counts",
            target_sum=1e4,
            make_backup=True,             # makes .bak once
            force_normalize_total=False,  # only normalize if totals don't already look normalized
        )

        print("\n=== Processed ===")
        print(f"path: {saved_path}")
        print(f"action: {report.get('action')}")
        print(f"used_counts_layer_as_source: {report.get('used_counts_layer_as_source')}")
        print(f"inference: {report.get('inference')}")
        print(f"source_summary(max/min/dtype): {report['source_summary']['maximum']:.4g} / "
              f"{report['source_summary']['minimum']:.4g} / {report['source_summary']['dtype']}")
        if "backup_path" in report:
            print(f"backup: {report['backup_path']}")