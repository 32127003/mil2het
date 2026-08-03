"""
example usage:
python scripts/preselection.py \
  --adata-path data.h5ad \
  --label-column phenotype \
  --celltype-column cell_type \
  --ppi-path ppi.tsv \
  --splits-directory outputs/splits \
  --output-dir outputs/preselection

"""
import argparse
import sys
import os
import re
from pathlib import Path

import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mil2het.config import parse_workflow_cli_args
from modules.utils import *


DEFAULT_DEG_MAX_P_VALUE = 0.05
DEFAULT_DEG_MIN_ABS_LOGFC = 1.0



def read_gene_list(file_path: str) -> List[str]:

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Gene list not found: {file_path}")

    dataframe = pd.read_csv(file_path, sep=None, engine="python")
    if dataframe.shape[1] == 0:
        return []

    for candidate_column in ["gene", "Gene", "GENE", "genes", "Genes", "symbol", "Symbol"]:
        if candidate_column in dataframe.columns:
            return dataframe[candidate_column].astype(str).tolist()

    first_column = dataframe.columns[0]
    return dataframe[first_column].astype(str).tolist()


def write_gene_list(
    file_path: str,
    genes: Sequence[str],
    scores: Optional[Sequence[float]] = None,
    extra_columns: Optional[Dict[str, Sequence]] = None,
) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if scores is not None and len(scores) != len(genes):
        raise ValueError("scores length must match genes length")

    output: Dict[str, Sequence] = {"gene": list(genes)}
    if scores is not None:
        output["score"] = list(scores)

    if extra_columns:
        for column_name, values in extra_columns.items():
            if len(values) != len(genes):
                raise ValueError(f"extra column '{column_name}' length must match genes length")
            output[column_name] = list(values)

    pd.DataFrame(output).to_csv(file_path, sep="\t", index=False)


def deduplicate_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        item_str = str(item)
        if item_str not in seen:
            seen.add(item_str)
            output.append(item_str)
    return output


def load_ppi_node_set(ppi_network_path: str) -> set:
    if not os.path.exists(ppi_network_path):
        raise FileNotFoundError(f"PPI network not found: {ppi_network_path}")

    node_set: set = set()
    with open(ppi_network_path, "r") as file_handle:
        for raw_line in file_handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                continue

            node_set.add(str(parts[0]))
            node_set.add(str(parts[1]))

    if len(node_set) == 0:
        raise RuntimeError(f"PPI network seems empty (no parsed edges): {ppi_network_path}")

    return node_set


def _clean_gene_identifier(value: object) -> str:
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return ""
    return text


def _overlap_count_with_ppi(genes: Sequence[str], ppi_node_set: set) -> int:
    clean_set = {g for g in (_clean_gene_identifier(value) for value in genes) if g != ""}
    return int(len(clean_set & set(ppi_node_set)))


def remap_var_names_to_best_symbol_source(
    adata: sc.AnnData,
    ppi_node_set: set,
    *,
    log_prefix: str = "",
) -> Tuple[sc.AnnData, Dict[str, object]]:
    """Remap adata.var_names to a var column with best overlap to PPI nodes.

    This handles datasets where var_names are Ensembl IDs while PPI uses symbols.
    """
    current_names = [str(value) for value in adata.var_names.tolist()]
    current_overlap = _overlap_count_with_ppi(current_names, ppi_node_set)

    # Keep original var_names namespace whenever it already overlaps PPI.
    # This avoids downstream mismatch with train.py, which indexes genes by adata.var_names.
    if int(current_overlap) > 0:
        print(
            f"{log_prefix}[gene-map] keep var_names (ppi overlap={int(current_overlap)}; remap skipped)",
            flush=True,
        )
        return adata, {
            "source": "var_names",
            "overlap_before": int(current_overlap),
            "overlap_after": int(current_overlap),
        }

    preferred_columns = [
        "feature_name",
        "gene_symbol",
        "symbol",
        "gene_name",
        "gene",
        "feature_id",
    ]
    keyword_columns = [
        column_name
        for column_name in adata.var.columns
        if any(keyword in str(column_name).lower() for keyword in ["gene", "symbol", "feature", "name"])
    ]

    candidates: List[str] = []
    for column_name in preferred_columns + keyword_columns:
        if column_name in adata.var.columns and column_name not in candidates:
            candidates.append(column_name)

    best_column: Optional[str] = None
    best_overlap = int(current_overlap)
    best_values: Optional[List[str]] = None
    for column_name in candidates:
        values = [str(value) for value in adata.var[column_name].tolist()]
        overlap = _overlap_count_with_ppi(values, ppi_node_set)
        if overlap > best_overlap:
            best_overlap = int(overlap)
            best_column = str(column_name)
            best_values = values

    if best_column is None or best_values is None:
        print(
            f"{log_prefix}[gene-map] keep var_names (ppi overlap={int(current_overlap)})",
            flush=True,
        )
        return adata, {
            "source": "var_names",
            "overlap_before": int(current_overlap),
            "overlap_after": int(current_overlap),
        }

    # Apply remap in-place on split-local adata object.
    if "var_name_original" not in adata.var.columns:
        adata.var["var_name_original"] = pd.Index(adata.var_names).astype(str)

    remapped = []
    for old_name, candidate_name in zip(adata.var_names.tolist(), best_values):
        candidate_clean = _clean_gene_identifier(candidate_name)
        if candidate_clean == "":
            remapped.append(str(old_name))
        else:
            remapped.append(str(candidate_clean))
    adata.var_names = pd.Index(remapped)
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()

    after_overlap = _overlap_count_with_ppi([str(value) for value in adata.var_names.tolist()], ppi_node_set)
    print(
        f"{log_prefix}[gene-map] var_names <- var['{best_column}'] "
        f"(ppi overlap {int(current_overlap)} -> {int(after_overlap)})",
        flush=True,
    )
    return adata, {
        "source": str(best_column),
        "overlap_before": int(current_overlap),
        "overlap_after": int(after_overlap),
    }


def load_split_train_indices(splits_dir: str, dataset: str, split_idx: int) -> np.ndarray:
    split_path = os.path.join(str(splits_dir), f"{str(dataset)}_idx_{int(split_idx)}.pkl")
    if not os.path.isfile(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with open(split_path, "rb") as file_handle:
        split_payload = pickle.load(file_handle)
    if not isinstance(split_payload, list) or len(split_payload) != 3:
        raise ValueError(f"Invalid split payload in {split_path}: expected list [train,val,test].")
    train_indices = np.asarray(split_payload[0], dtype=np.int64)
    if train_indices.size == 0:
        raise ValueError(f"Train split is empty: {split_path}")
    return train_indices


def infer_dataset_name(config) -> str:
    dataset_value = str(getattr(config, "dataset", "") or "").strip()
    if dataset_value != "":
        return dataset_value

    adata_path = str(
        getattr(config, "adata_path", "") or getattr(config, "input_h5ad", "") or ""
    ).strip()
    if adata_path == "":
        raise ValueError("Either config.dataset or config.adata_path/input_h5ad must be set.")

    stem = Path(adata_path).stem
    if stem.endswith("_data"):
        stem = stem[: -len("_data")]
    if stem == "":
        raise ValueError(f"Could not infer dataset name from adata path: {adata_path}")
    return stem


def resolve_adata_path(config) -> str:
    adata_path = str(
        getattr(config, "adata_path", "") or getattr(config, "input_h5ad", "") or ""
    ).strip()
    if adata_path != "":
        return adata_path

    dataset_name = infer_dataset_name(config)
    adata_directory = str(getattr(config, "adata_directory", "") or "").strip()
    if adata_directory == "":
        raise ValueError("Either config.adata_path/input_h5ad or config.adata_directory must be set.")
    return os.path.join(adata_directory, dataset_name, f"{dataset_name}_data.h5ad")


def resolve_preselection_output_root(config) -> str:
    output_root = str(getattr(config, "preselection_output_root", "") or "").strip()
    if output_root == "":
        raise ValueError("config.preselection_output_root must be set.")
    return output_root


def discover_split_indices(splits_dir: str, dataset_name: str, num_folds: Optional[int] = None) -> List[int]:
    pattern = re.compile(rf"^{re.escape(str(dataset_name))}_idx_(\d+)\.pkl$")
    discovered: List[int] = []
    for file_name in os.listdir(splits_dir):
        match = pattern.match(str(file_name))
        if match is None:
            continue
        discovered.append(int(match.group(1)))

    if len(discovered) > 0:
        return sorted(set(discovered))
    if num_folds is not None:
        return [int(index) for index in range(int(num_folds))]
    raise FileNotFoundError(
        f"No split files found under {splits_dir} matching {dataset_name}_idx_<n>.pkl."
    )


############################################################ (for DEG analysis) ##################################################################




def collect_deg_rows(
    adata: sc.AnnData,
    groupby: str,
    groups: Sequence[str],
    ppi_node_set: set,
    *,
    max_p_value: float,
    min_abs_logfc: float,
    apply_thresholds: bool = True,
) -> pd.DataFrame:
    if groupby not in adata.obs.columns:
        raise KeyError(f"groupby column '{groupby}' not found in adata.obs")

    adata_tmp = adata.copy()
    adata_tmp.obs[groupby] = adata_tmp.obs[groupby].astype(str).astype("category")
    group_values = adata_tmp.obs[groupby].astype(str)
    total_cells = int(group_values.shape[0])
    groups_str = [str(g) for g in groups]
    available_groups = set(map(str, adata_tmp.obs[groupby].cat.categories))

    collected_rows: List[Dict[str, object]] = []

    for group in groups_str:
        if group not in available_groups:
            continue
        group_cells = int((group_values == str(group)).sum())
        rest_cells = int(total_cells - group_cells)
        if group_cells < 10 or rest_cells < 10:
            continue
        try:
            sc.tl.rank_genes_groups(
                adata_tmp,
                groupby=groupby,
                groups=[group],
                reference="rest",
                method="t-test",
                use_raw=False,
            )
        except ZeroDivisionError:
            continue
        result = adata_tmp.uns.get("rank_genes_groups", None)
        if result is None:
            raise RuntimeError("Scanpy did not populate adata.uns['rank_genes_groups']")

        names = np.array(result["names"][group], dtype=str)

        if "pvals_adj" in result:
            pvals = np.array(result["pvals_adj"][group], dtype=np.float64)
        elif "pvals" in result:
            pvals = np.array(result["pvals"][group], dtype=np.float64)
        else:
            pvals = np.ones(len(names), dtype=np.float64)

        if "logfoldchanges" in result:
            logfc = np.array(result["logfoldchanges"][group], dtype=np.float64)
        else:
            logfc = np.zeros(len(names), dtype=np.float64)

        df = pd.DataFrame({"gene": names, "pval": pvals, "logfc": logfc, "group": group})
        df = df[df["gene"].isin(ppi_node_set)].copy()

        df["pval"] = pd.to_numeric(df["pval"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        df["logfc"] = pd.to_numeric(df["logfc"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["pval", "logfc"])

        if apply_thresholds:
            df = df[(df["pval"] <= float(max_p_value)) & (df["logfc"].abs() >= float(min_abs_logfc))].copy()
            if df.empty:
                continue

        collected_rows.extend(df.to_dict(orient="records"))

    if len(collected_rows) == 0:
        return pd.DataFrame(columns=["gene", "pval", "logfc", "group"])

    return pd.DataFrame(collected_rows)


def deg_rank_by_abs_logfc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    ranked = df.copy()
    ranked["abs_logfc"] = ranked["logfc"].abs()
    ranked = ranked.sort_values(["abs_logfc", "pval"], ascending=[False, True])
    ranked = ranked.drop_duplicates(subset=["gene"], keep="first").copy()
    return ranked


def deg_metrics_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["gene", "pval", "logfc", "abs_logfc", "regulation"])
    abs_logfc = df["abs_logfc"] if "abs_logfc" in df.columns else df["logfc"].abs()
    return pd.DataFrame(
        {
            "gene": df["gene"].astype(str),
            "pval": df["pval"].astype(float),
            "logfc": df["logfc"].astype(float),
            "abs_logfc": abs_logfc.astype(float),
            "regulation": [1 if float(x) > 0.0 else -1 for x in df["logfc"]],
        }
    )


def select_deg(
    adata: sc.AnnData,
    groupby: str,
    groups: Sequence[str],
    ppi_node_set: set,
    *,
    max_p_value: float = 0.05,
    min_abs_logfc: float = 1.0,
) -> Tuple[
    List[str], List[float], List[float],   # pass_genes, pass_pvals, pass_logfc
]:
    pass_df = collect_deg_rows(
        adata=adata,
        groupby=groupby,
        groups=groups,
        ppi_node_set=ppi_node_set,
        max_p_value=max_p_value,
        min_abs_logfc=min_abs_logfc,
        apply_thresholds=True,
    )

    if pass_df.empty:
        raise RuntimeError(
            f"DEG pass set is empty under thresholds: p<={max_p_value}, |logfc|>={min_abs_logfc}. "
            "Loosen thresholds or check group labels."
        )


    pass_df = deg_rank_by_abs_logfc(pass_df)

    pass_genes = pass_df["gene"].astype(str).tolist()
    pass_pvals = pass_df["pval"].astype(float).tolist()
    pass_logfc = pass_df["logfc"].astype(float).tolist()

    return (
        pass_genes, pass_pvals, pass_logfc,
    )


def select_deg_union_by_celltype(
    adata: sc.AnnData,
    celltype_column: str,
    label_column: str,
    label_groups: Sequence[str],
    ppi_node_set: set,
    *,
    max_p_value: float = 0.05,
    min_abs_logfc: float = 1.0,
    allow_empty_pass_set: bool = False,
) -> Tuple[
    List[str], List[float], List[float],
    Dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, List[str]],
    Dict[str, Dict[str, pd.DataFrame]],
]:
    if celltype_column not in adata.obs.columns:
        raise KeyError(f"celltype_column '{celltype_column}' not found in adata.obs")
    if label_column not in adata.obs.columns:
        raise KeyError(f"label_column '{label_column}' not found in adata.obs")

    label_groups_str = [str(g) for g in label_groups]
    celltype_values = adata.obs[celltype_column].astype(str)
    celltype_list = [ct for ct in pd.unique(celltype_values) if ct and ct.lower() not in {"nan", "none"}]

    collected_pass: List[pd.DataFrame] = []
    skipped: List[str] = []
    per_celltype: Dict[str, Dict[str, pd.DataFrame]] = {}

    for celltype in celltype_list:
        mask = celltype_values == celltype
        adata_ct = adata[mask].copy()
        label_counts = (
            adata_ct.obs[label_column]
            .astype(str)
            .value_counts()
        )
        label_values = set(map(str, label_counts.index.tolist()))
        if not all(group in label_values for group in label_groups_str):
            skipped.append(celltype)
            empty_df = pd.DataFrame(columns=["gene", "pval", "logfc", "group"])
            per_celltype[celltype] = {"max_df": empty_df, "pass_df": empty_df.copy()}
            continue
        if any(int(label_counts.get(group, 0)) < 2 for group in label_groups_str):
            skipped.append(celltype)
            empty_df = pd.DataFrame(columns=["gene", "pval", "logfc", "group"])
            per_celltype[celltype] = {"max_df": empty_df, "pass_df": empty_df.copy()}
            continue

        all_df = collect_deg_rows(
            adata=adata_ct,
            groupby=label_column,
            groups=label_groups_str,
            ppi_node_set=ppi_node_set,
            max_p_value=1.0,
            min_abs_logfc=0.0,
            apply_thresholds=False,
        )
        if all_df.empty:
            per_celltype[celltype] = {"max_df": all_df.copy(), "pass_df": all_df.copy()}
            continue

        max_df = deg_rank_by_abs_logfc(all_df)
        pass_df = all_df[
            (all_df["pval"] <= float(max_p_value)) & (all_df["logfc"].abs() >= float(min_abs_logfc))
        ].copy()
        if not pass_df.empty:
            pass_df = deg_rank_by_abs_logfc(pass_df)

        per_celltype[celltype] = {"max_df": max_df, "pass_df": pass_df}

        if not pass_df.empty:
            pass_df = pass_df.copy()
            pass_df["celltype"] = celltype
            collected_pass.append(pass_df)

    if len(collected_pass) == 0:
        if not bool(allow_empty_pass_set):
            raise RuntimeError(
                f"DEG pass set is empty across celltypes under thresholds: p<={max_p_value}, "
                f"|logfc|>={min_abs_logfc}. Check label groups and celltype coverage."
            )
        summary = {
            "celltypes_total": int(len(celltype_list)),
            "celltypes_used": int(len(celltype_list) - len(skipped)),
            "celltypes_skipped": skipped,
        }
        return (
            [], [], [],
            summary,
            pd.DataFrame(columns=["celltype", "deg_count"]),
            pd.DataFrame(columns=["overlap_celltypes", "gene_count"]),
            {},
            per_celltype,
        )

    pass_df = pd.concat(collected_pass, ignore_index=True)
    unique_celltype_genes = pass_df.drop_duplicates(subset=["celltype", "gene"])[["celltype", "gene"]]
    celltype_counts_df = (
        unique_celltype_genes.groupby("celltype")["gene"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"gene": "deg_count"})
    )
    overlap_series = unique_celltype_genes.groupby("gene")["celltype"].nunique()
    overlap_counts_df = overlap_series.value_counts().sort_index().reset_index()
    overlap_counts_df.columns = ["overlap_celltypes", "gene_count"]
    gene_celltype_map = (
        unique_celltype_genes.groupby("gene")["celltype"]
        .apply(lambda series: sorted(set(series.astype(str).tolist())))
        .to_dict()
    )
    pass_df = deg_rank_by_abs_logfc(pass_df)

    pass_genes = pass_df["gene"].astype(str).tolist()
    pass_pvals = pass_df["pval"].astype(float).tolist()
    pass_logfc = pass_df["logfc"].astype(float).tolist()

    summary = {
        "celltypes_total": int(len(celltype_list)),
        "celltypes_used": int(len(celltype_list) - len(skipped)),
        "celltypes_skipped": skipped,
    }

    return (
        pass_genes, pass_pvals, pass_logfc,
        summary,
        celltype_counts_df,
        overlap_counts_df,
        gene_celltype_map,
        per_celltype,
    )


def compute_global_deg_zscore(
    adata: sc.AnnData,
    groupby: str,
    positive_label: str,
    negative_label: str,
    ppi_node_set: set,
    *,
    method: str = "wilcoxon",
) -> pd.DataFrame:
    """Compute a global DEG statistic (scanpy rank_genes_groups `scores`) for all genes.

    Notes
    - For `method='wilcoxon'`, scanpy's `scores` are z-scores.
    - We store the output as `zscore` regardless of method for convenience.
    """
    if groupby not in adata.obs.columns:
        raise KeyError(f"groupby column '{groupby}' not found in adata.obs")

    adata_tmp = adata.copy()
    adata_tmp.obs[groupby] = adata_tmp.obs[groupby].astype(str).astype("category")
    positive_label = str(positive_label)
    negative_label = str(negative_label)
    categories = set(map(str, adata_tmp.obs[groupby].cat.categories))
    if positive_label not in categories or negative_label not in categories:
        raise ValueError(
            f"Labels '{positive_label}'/'{negative_label}' not found in {groupby} categories={sorted(categories)}"
        )

    sc.tl.rank_genes_groups(
        adata_tmp,
        groupby=groupby,
        groups=[positive_label],
        reference=negative_label,
        method=str(method),
        n_genes=int(adata_tmp.n_vars),
        use_raw=False,
    )
    result = adata_tmp.uns.get("rank_genes_groups", None)
    if result is None:
        raise RuntimeError("Scanpy did not populate adata.uns['rank_genes_groups']")

    names = np.array(result["names"][positive_label], dtype=str)
    scores = np.array(result["scores"][positive_label], dtype=np.float64)

    if "pvals_adj" in result:
        pvals = np.array(result["pvals_adj"][positive_label], dtype=np.float64)
    elif "pvals" in result:
        pvals = np.array(result["pvals"][positive_label], dtype=np.float64)
    else:
        pvals = np.ones(len(names), dtype=np.float64)

    if "logfoldchanges" in result:
        logfc = np.array(result["logfoldchanges"][positive_label], dtype=np.float64)
    else:
        logfc = np.zeros(len(names), dtype=np.float64)

    df = pd.DataFrame(
        {
            "gene": [str(g).upper() for g in names],
            "zscore": scores.astype(float),
            "pval": pvals.astype(float),
            "logfc": logfc.astype(float),
        }
    )
    df = df[df["gene"].isin({str(g) for g in ppi_node_set})].copy()
    df = df.dropna(subset=["zscore"]).copy()
    df = df.sort_values("zscore", ascending=False)
    df = df.drop_duplicates(subset=["gene"], keep="first").reset_index(drop=True)
    return df



############################################################ (for Network Propagation) ##################################################################


@dataclass(frozen=True)
class RWRConfig:
    restart_probability: float = 0.1
    convergence_threshold_l1: float = 1e-6
    max_iterations: int = 1000
    treat_as_undirected: bool = True 


def parse_ppi_edges_and_nodes_in_adata_namespace(
    edge_list_path: str,
    allowed_gene_set: set,
) -> Tuple[List[str], List[Tuple[str, str]]]:

    node_order: List[str] = []
    node_seen: set = set()
    edges: List[Tuple[str, str]] = []

    with open(edge_list_path, "r") as file_handle:
        for raw_line in file_handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                continue

            gene_u = str(parts[0])
            gene_v = str(parts[1])

            if gene_u not in allowed_gene_set or gene_v not in allowed_gene_set:
                continue

            if gene_u not in node_seen:
                node_seen.add(gene_u)
                node_order.append(gene_u)
            if gene_v not in node_seen:
                node_seen.add(gene_v)
                node_order.append(gene_v)

            edges.append((gene_u, gene_v))

    return node_order, edges


def build_row_normalized_transposed_adjacency(
    node_order: Sequence[str],
    edges: Sequence[Tuple[str, str]],
    treat_as_undirected: bool,
) -> sparse.csr_matrix:

    node_to_index: Dict[str, int] = {str(g): i for i, g in enumerate(node_order)}
    n = len(node_order)

    row_indices: List[int] = []
    col_indices: List[int] = []
    data: List[float] = []

    for gene_u, gene_v in edges:
        index_u = node_to_index[gene_u]
        index_v = node_to_index[gene_v]

        row_indices.append(index_u)  # src
        col_indices.append(index_v)  # dst
        data.append(1.0)

        if treat_as_undirected:
            row_indices.append(index_v)
            col_indices.append(index_u)
            data.append(1.0)

    adjacency = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n, n), dtype=np.float64)

    adjacency_t = adjacency.transpose().tocsr()  # A^T
    row_sums = np.asarray(adjacency_t.sum(axis=1)).ravel()
    row_sums[row_sums == 0.0] = 1.0
    inv_row_sums = 1.0 / row_sums

    p_matrix = sparse.diags(inv_row_sums, offsets=0, format="csr") @ adjacency_t
    return p_matrix


def seed_vector(
    seed_genes: Sequence[str],
    node_to_index: Dict[str, int],
) -> np.ndarray:

    p0 = np.zeros(len(node_to_index), dtype=np.float64)
    for gene in seed_genes:
        idx = node_to_index.get(str(gene), None)
        if idx is None:
            continue
        p0[int(idx)] += 1.0

    if float(p0.sum()) <= 0.0:
        raise RuntimeError(
            "None of the seed genes overlap with the network nodes (after intersecting with adata.var_names). "
            "Check identifiers and the DEG seed file content."
        )
    # Keep raw seed weights (network_propagation.py default behavior).
    # Normalization can be added later as an option if needed.
    return p0


def run_rwr(
    p_matrix: sparse.csr_matrix,
    p0: np.ndarray,
    rwr_config: RWRConfig,
) -> np.ndarray:
    """
    Update:
      p_{t+1} = (1-r) * (P @ p_t) + r * p0
    where P = row-normalize(A^T)
    until convergence or max iterations.
    """
    p = p0.copy()

    for _ in range(int(rwr_config.max_iterations)):
        p_new = (1.0 - float(rwr_config.restart_probability)) * (p_matrix @ p) + float(rwr_config.restart_probability) * p0
        diff_norm = float(np.linalg.norm(p_new - p, ord=1))
        p = np.asarray(p_new).ravel()
        if diff_norm < float(rwr_config.convergence_threshold_l1):
            break

    return np.asarray(p).ravel()


def np_scores(
    adata: sc.AnnData,
    network_path: str,
    seed_genes: Sequence[str],
    rwr_config: RWRConfig,
) -> Tuple[List[str], List[float]]:
    if not os.path.exists(network_path):
        raise FileNotFoundError(f"PPI network not found: {network_path}")

    adata_gene_set = set(map(str, adata.var_names))

    node_order, edges = parse_ppi_edges_and_nodes_in_adata_namespace(
        edge_list_path=network_path,
        allowed_gene_set=adata_gene_set,
    )
    if len(node_order) == 0 or len(edges) == 0:
        raise RuntimeError(
            "No usable edges/nodes after restricting to adata.var_names. "
            "Check that adata.var_names uses the same identifiers as the PPI file."
        )

    node_to_index = {gene: i for i, gene in enumerate(node_order)}

    seed_genes_in_graph = [g for g in seed_genes if str(g) in node_to_index]
    p0 = seed_vector(seed_genes_in_graph, node_to_index=node_to_index)

    p_matrix = build_row_normalized_transposed_adjacency(
        node_order=node_order,
        edges=edges,
        treat_as_undirected=bool(rwr_config.treat_as_undirected),
    )

    scores = run_rwr(p_matrix=p_matrix, p0=p0, rwr_config=rwr_config)

    order = np.argsort(-scores)
    genes_sorted = [node_order[int(i)] for i in order]
    scores_sorted = [float(scores[int(i)]) for i in order]
    return genes_sorted, scores_sorted



################################################################## main run ##################################################################

def preselection(
    adata: sc.AnnData,
    ppi_network_path: str,
    groupby: str,
    group1: str,
    group2: str,
    celltype_column: str,
    output_dir: str,
    split_idx: Optional[int] = None,
    deg_max_p_value: Optional[float] = None,
    deg_min_abs_logfc: Optional[float] = None,
    restart_prob: Optional[float] = None,
    convergence_threshold_l1: Optional[float] = None,
    max_iterations: Optional[int] = None,
    directed: Optional[bool] = None,
) -> None:
    
    if not ppi_network_path:
        raise ValueError("provide ppi_network path in config for preselection step")

    prefix = f"[split {int(split_idx)}] " if split_idx is not None else ""
    ppi_node_set = load_ppi_node_set(ppi_network_path)
    adata, gene_map_info = remap_var_names_to_best_symbol_source(
        adata=adata,
        ppi_node_set=ppi_node_set,
        log_prefix=prefix,
    )
    if int(gene_map_info.get("overlap_after", 0)) == 0:
        print(
            f"{prefix}[warn] zero overlap between adata genes and PPI nodes after gene mapping. "
            "DEG/NP outputs may be empty.",
            flush=True,
        )
    
    #Gene space reduction with DEG analysis, by cell type. 
    deg_out_dir = os.path.join(output_dir, "DEG")
    os.makedirs(deg_out_dir, exist_ok=True)
    groups = [group1, group2]

    (
        deg_pass_genes, _deg_pass_pvals, _deg_pass_logfc,
        celltype_summary,
        celltype_counts_df,
        overlap_counts_df,
        gene_celltype_map,
        per_celltype_deg,
    ) = select_deg_union_by_celltype(
        adata=adata,
        celltype_column=celltype_column,
        label_column=str(groupby),
        label_groups=groups,
        ppi_node_set=ppi_node_set,
        max_p_value=float(
            deg_max_p_value if deg_max_p_value is not None else DEFAULT_DEG_MAX_P_VALUE
        ),
        min_abs_logfc=float(
            deg_min_abs_logfc if deg_min_abs_logfc is not None else DEFAULT_DEG_MIN_ABS_LOGFC
        ),
        allow_empty_pass_set="allow_empty",
    )
    for celltype_name, payload in per_celltype_deg.items():
        safe_celltype = sanitize_filename_component(celltype_name)
        max_df = payload.get("max_df", pd.DataFrame())
        pass_df = payload.get("pass_df", pd.DataFrame())

        metrics_max_path = os.path.join(deg_out_dir, f"DEG_metrics_{safe_celltype}_max.tsv")
        metrics_pass_path = os.path.join(deg_out_dir, f"DEG_metrics_{safe_celltype}_pass.tsv")
        genes_max_path = os.path.join(deg_out_dir, f"DEG_genes_{safe_celltype}_max.tsv")
        genes_pass_path = os.path.join(deg_out_dir, f"DEG_genes_{safe_celltype}_pass.tsv")

        deg_metrics_frame(max_df).to_csv(metrics_max_path, sep="\t", index=False)
        deg_metrics_frame(pass_df).to_csv(metrics_pass_path, sep="\t", index=False)

        write_gene_list(
            file_path=genes_max_path,
            genes=max_df["gene"].astype(str).tolist() if not max_df.empty else [],
        )
        write_gene_list(
            file_path=genes_pass_path,
            genes=pass_df["gene"].astype(str).tolist() if not pass_df.empty else [],
        )
    if celltype_summary.get("celltypes_total", 0) > 0:
        skipped = celltype_summary.get("celltypes_skipped", [])
        skipped_preview = skipped[:5]
        skipped_msg = f", skipped={len(skipped)}"
        if skipped_preview:
            skipped_msg += f" (e.g., {skipped_preview})"
        print(
            f"{prefix}[DEG] celltype-stratified selection: "
            f"celltype_column={celltype_column}, label_column={groupby}, groups={groups}, "
            f"celltypes_used={celltype_summary.get('celltypes_used')}/"
            f"{celltype_summary.get('celltypes_total')}{skipped_msg}",
            flush=True,
        )

        celltype_counts_path = os.path.join(deg_out_dir, "DEG_pass_celltype_counts.tsv")
        celltype_counts_df.to_csv(celltype_counts_path, sep="\t", index=False)
        print(f"{prefix}[DEG] celltype counts wrote: {celltype_counts_path}", flush=True)

        overlap_counts_path = os.path.join(deg_out_dir, "DEG_pass_overlap_counts.tsv")
        overlap_counts_df.to_csv(overlap_counts_path, sep="\t", index=False)
        print(f"{prefix}[DEG] overlap counts wrote: {overlap_counts_path}", flush=True)

        total_genes = int(overlap_counts_df["gene_count"].sum())
        total_selections = int(
            (overlap_counts_df["overlap_celltypes"] * overlap_counts_df["gene_count"]).sum()
        )
        overlap1_count = int(
            overlap_counts_df.loc[
                overlap_counts_df["overlap_celltypes"] == 1, "gene_count"
            ].sum()
        )
        overlap2plus_count = int(
            overlap_counts_df.loc[
                overlap_counts_df["overlap_celltypes"] >= 2, "gene_count"
            ].sum()
        )
        overlap3plus_count = int(
            overlap_counts_df.loc[
                overlap_counts_df["overlap_celltypes"] >= 3, "gene_count"
            ].sum()
        )

        if total_genes > 0:
            uniqueness_ratio = overlap1_count / float(total_genes)
            mean_overlap = total_selections / float(total_genes)
            overlap2plus_ratio = overlap2plus_count / float(total_genes)
            overlap3plus_ratio = overlap3plus_count / float(total_genes)
            p = overlap_counts_df["gene_count"].astype(float).to_numpy() / float(total_genes)
            entropy = float(-np.sum(p * np.log(p + 1e-12)))
        else:
            uniqueness_ratio = 0.0
            mean_overlap = 0.0
            overlap2plus_ratio = 0.0
            overlap3plus_ratio = 0.0
            entropy = 0.0

        celltype_counts = celltype_counts_df["deg_count"].astype(float).to_numpy()
        celltype_mean = float(celltype_counts.mean()) if celltype_counts.size > 0 else 0.0
        celltype_std = float(celltype_counts.std(ddof=0)) if celltype_counts.size > 0 else 0.0
        celltype_cv = float(celltype_std / celltype_mean) if celltype_mean > 0 else 0.0

        overlap_metrics = pd.DataFrame(
            [
                {"metric": "total_genes", "value": total_genes},
                {"metric": "total_selections", "value": total_selections},
                {"metric": "uniqueness_ratio", "value": uniqueness_ratio},
                {"metric": "mean_overlap", "value": mean_overlap},
                {"metric": "overlap2plus_ratio", "value": overlap2plus_ratio},
                {"metric": "overlap3plus_ratio", "value": overlap3plus_ratio},
                {"metric": "overlap_entropy", "value": entropy},
                {"metric": "celltype_count_mean", "value": celltype_mean},
                {"metric": "celltype_count_std", "value": celltype_std},
                {"metric": "celltype_count_cv", "value": celltype_cv},
                {"metric": "celltypes_used", "value": celltype_summary.get("celltypes_used")},
                {"metric": "celltypes_total", "value": celltype_summary.get("celltypes_total")},
            ]
        )
        overlap_metrics_path = os.path.join(deg_out_dir, "DEG_pass_overlap_metrics.tsv")
        overlap_metrics.to_csv(overlap_metrics_path, sep="\t", index=False)
        print(f"{prefix}[DEG] overlap metrics wrote: {overlap_metrics_path}", flush=True)
    
    deg_merged_path = os.path.join(deg_out_dir, "DEG_merged.tsv")
    write_gene_list(file_path=deg_merged_path, genes=deduplicate_preserve_order(deg_pass_genes))
    print(f"{prefix}[DEG] merged genes wrote: {deg_merged_path}")
    if len(groups) >= 2:
        zscore_df = compute_global_deg_zscore(
            adata=adata,
            groupby=str(groupby),
            positive_label=str(groups[0]),
            negative_label=str(groups[1]),
            ppi_node_set=ppi_node_set,
            method="wilcoxon",
        )
        zscore_path = os.path.join(deg_out_dir, "DEG_zscore_global.tsv")
        zscore_df.to_csv(zscore_path, sep="\t", index=False)
        print(f"{prefix}[DEG] global zscore wrote: {zscore_path} (n={int(zscore_df.shape[0])})", flush=True)
    else:
        print(f"{prefix}[warn] skipping global DEG zscore; need at least 2 groups (pos,neg).", flush=True)
    

    #Gene space extension with Network Propagation. Use top k NP ranked genes later for training
    
    
    np_out_dir = os.path.join(output_dir, "NP")
    os.makedirs(np_out_dir, exist_ok=True)

    deg_merged_path = os.path.join(deg_out_dir, "DEG_merged.tsv")
    seed_genes_for_np = deduplicate_preserve_order(read_gene_list(deg_merged_path))
    print(f"{prefix}[NP] running RWR with {len(seed_genes_for_np)} seed genes from {deg_merged_path}", flush=True)

    def write_empty_np_max(reason: str) -> None:
        np_max_path = os.path.join(np_out_dir, "NP_max.tsv")
        write_gene_list(
            file_path=np_max_path,
            genes=[],
            scores=[],
            extra_columns={"deg_seed_nonoverlap": []},
        )
        print(f"{prefix}[warn] NP skipped: {reason}", flush=True)
        print(f"{prefix}[NP] wrote: {np_max_path} (n=0)", flush=True)

    if len(seed_genes_for_np) == 0:
        write_empty_np_max("no DEG seeds available")
        return

    rwr_defaults = RWRConfig()
    rwr_config = RWRConfig(
        restart_probability=float(
            restart_prob if restart_prob is not None else rwr_defaults.restart_probability
        ),
        convergence_threshold_l1=float(
            convergence_threshold_l1
            if convergence_threshold_l1 is not None
            else rwr_defaults.convergence_threshold_l1
        ),
        max_iterations=int(
            max_iterations if max_iterations is not None else rwr_defaults.max_iterations
        ),
        treat_as_undirected=(
            not bool(directed)
            if directed is not None
            else bool(rwr_defaults.treat_as_undirected)
        ),
    )

    try:
        np_genes_sorted, np_score_values_sorted = np_scores(
            adata=adata,
            network_path=str(ppi_network_path),
            seed_genes=seed_genes_for_np,
            rwr_config=rwr_config,
        )
    except RuntimeError as error:
        if "None of the seed genes overlap with the network nodes" not in str(error):
            raise
        write_empty_np_max("seed genes do not overlap PPI nodes")
        return
    seed_set = set(map(str, seed_genes_for_np))

    def write_np_file(path: str, records: Sequence[Tuple[str, float]]) -> None:
        genes_ordered = [gene for gene, _ in records]
        scores_ordered = [score for _, score in records]
        nonoverlap = [0 if gene in seed_set else 1 for gene in genes_ordered]
        write_gene_list(
            file_path=path,
            genes=genes_ordered,
            scores=scores_ordered,
            extra_columns={"deg_seed_nonoverlap": nonoverlap},
        )

    full_records: List[Tuple[str, float]] = [
        (str(gene), float(score))
        for gene, score in zip(np_genes_sorted, np_score_values_sorted)
    ]

    np_max_path = os.path.join(np_out_dir, "NP_max.tsv")
    write_np_file(np_max_path, full_records)
    print(f"{prefix}[NP] wrote: {np_max_path} (n={len(full_records)})")


def run_split_preselection(config) -> str:
    if not getattr(config, "splits_directory", None):
        raise ValueError("splits_directory must be specified in the configuration.")
    if not getattr(config, "ppi_path", None):
        raise ValueError("ppi_path must be specified in the configuration.")

    dataset_name = infer_dataset_name(config)
    adata_path = resolve_adata_path(config)
    print(f"reading adata from {adata_path}...")
    adata = sc.read_h5ad(adata_path)

    preselection_output_root = resolve_preselection_output_root(config)
    os.makedirs(preselection_output_root, exist_ok=True)
    split_indices = discover_split_indices(
        splits_dir=str(config.splits_directory),
        dataset_name=dataset_name,
        num_folds=getattr(config, "num_folds", None),
    )

    for split_idx in split_indices:
        train_indices = load_split_train_indices(
            splits_dir=str(config.splits_directory),
            dataset=dataset_name,
            split_idx=split_idx,
        )
        adata_train = adata[train_indices].copy()
        split_out_dir = os.path.join(preselection_output_root, f"split_{split_idx}")
        os.makedirs(split_out_dir, exist_ok=True)
        print(
            f"Running preselection for split {split_idx} with {len(train_indices)} training samples",
            flush=True,
        )
        preselection(
            adata=adata_train,
            ppi_network_path=str(config.ppi_path),
            groupby=str(config.label_column),
            group1=str(getattr(config, "binary_positive_label", "1")),
            group2=str(getattr(config, "binary_negative_label", "0")),
            celltype_column=str(config.celltype_column),
            output_dir=split_out_dir,
            split_idx=split_idx,
            deg_max_p_value=float(getattr(config, "deg_max_p_value", 0.05)),
            deg_min_abs_logfc=float(getattr(config, "deg_min_abs_logfc", 1.0)),
            restart_prob=float(getattr(config, "restart_prob", 0.1)),
            convergence_threshold_l1=float(getattr(config, "convergence_threshold_l1", 1e-6)),
            max_iterations=int(getattr(config, "max_iterations", 1000)),
            directed=bool(getattr(config, "directed", False)),
        )

    print(f"Split output directory: {preselection_output_root}", flush=True)
    return preselection_output_root


def build_preselection_config_from_cli_args(args):
    if args.adata_path is None:
        raise ValueError("--adata-path is required.")
    if args.label_column is None:
        raise ValueError("--label-column is required.")
    if args.celltype_column is None:
        raise ValueError("--celltype-column is required.")
    if args.splits_directory is None:
        raise ValueError("--splits-directory is required.")
    if args.ppi_path is None:
        raise ValueError("--ppi-path is required.")
    if args.output_dir is None:
        raise ValueError("--output-dir is required.")

    dataset_name = str(args.dataset_name or "").strip()
    if dataset_name == "":
        stem = Path(str(args.adata_path)).stem
        dataset_name = stem[: -len("_data")] if stem.endswith("_data") else stem

    config_dict = {
        "adata_path": str(args.adata_path),
        "dataset": str(dataset_name),
        "celltype_column": str(args.celltype_column),
        "label_column": str(args.label_column),
        "ppi_path": str(args.ppi_path),
        "splits_directory": str(args.splits_directory),
        "preselection_output_root": str(args.output_dir),
        "num_folds": int(args.num_folds) if args.num_folds is not None else 5,
        "binary_positive_label": str(args.positive_label or "1"),
        "binary_negative_label": str(args.negative_label or "0"),
        "deg_max_p_value": float(args.deg_max_p_value if args.deg_max_p_value is not None else 0.05),
        "deg_min_abs_logfc": float(args.deg_min_abs_logfc if args.deg_min_abs_logfc is not None else 1.0),
        "restart_prob": float(args.restart_prob if args.restart_prob is not None else 0.1),
        "convergence_threshold_l1": float(
            args.convergence_threshold_l1 if args.convergence_threshold_l1 is not None else 1e-6
        ),
        "max_iterations": int(args.max_iterations if args.max_iterations is not None else 1000),
        "directed": bool(args.directed),
    }
    return dict2namespace(config_dict)


def build_preselection_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split-aware preselection for DEG/NP (split-train-only).")
    parser.add_argument("--dataset-name", type=str, default=None, help="Explicit dataset/output prefix.")
    parser.add_argument("--adata-path", type=str, default=None, help="Explicit .h5ad path.")
    parser.add_argument("--label-column", type=str, default=None, help="Label column in adata.obs.")
    parser.add_argument("--celltype-column", type=str, default=None, help="Cell type column in adata.obs.")
    parser.add_argument("--ppi-path", type=str, default=None, help="PPI network path.")
    parser.add_argument("--splits-directory", type=str, default=None, help="Directory containing split pickle files.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for preselection artifacts.")
    parser.add_argument("--num-folds", type=int, default=None)
    parser.add_argument("--positive-label", type=str, default=None)
    parser.add_argument("--negative-label", type=str, default=None)
    parser.add_argument("--deg-max-p-value", type=float, default=None)
    parser.add_argument("--deg-min-abs-logfc", type=float, default=None)
    parser.add_argument("--restart-prob", type=float, default=None)
    parser.add_argument("--convergence-threshold-l1", type=float, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--directed", action="store_true")
    return parser


def load_preselection_config_from_cli(argv: Optional[Sequence[str]] = None):
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if "--config" in argv_list or "--adata" in argv_list:
        return parse_workflow_cli_args(argv_list).config

    parser = build_preselection_arg_parser()
    args = parser.parse_args(argv_list)
    return build_preselection_config_from_cli_args(args)







if __name__ == "__main__":
    config = load_preselection_config_from_cli()
    print(f"Configuration: {config}")
    run_split_preselection(config)
