"""Evaluate biomarker stability across folds and optional DEG agreement.

This script is evaluation-only. It does not train models.

Expected biomarker inputs per fold (either format):
- Current: `biomarker/final_biomarker.tsv` with columns `gene, S_final`
- Legacy: `biomarker/gene_biomarker/gene_importance_{fold}.tsv` with columns `gene_symbol, score`
- Optional legacy edges: `biomarker/network_biomarker/network_edges_*_fold{fold}.tsv`
  with columns `gene_1, gene_2, score` (extra columns are ignored)

Folder layouts supported:
- Legacy: `<experiment_root>/fold_{fold}_{seed}/...`
- Current: `<experiment_root>/split_idx_{fold}/seed_{seed}/...`
- Signature: `<experiment_root>/{prototype_prior_enabled}_{protein_embedding_merge}_{sample_bags}_{pooling}/split_idx_{fold}/seed_{seed}/...`
"""

import argparse
import importlib.util
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt


def safe_string_for_filename(text: str) -> str:
    return str(text).replace("/", "_").replace(" ", "_")


def load_config_dictionary(config_path: str) -> Dict:
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config_dict = getattr(module, "config", None)
    if config_dict is None or not isinstance(config_dict, dict):
        raise ValueError("Config file must define dictionary variable `config`.")
    return config_dict


def read_gene_ranking_table(path: str) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    expected_columns = {"gene_symbol", "score"}
    missing_columns = expected_columns.difference(set(table.columns))
    if missing_columns:
        raise ValueError(f"Gene table missing columns {sorted(missing_columns)}: {path}")
    table = table[["gene_symbol", "score"]].copy()
    table["gene_symbol"] = table["gene_symbol"].astype(str)
    table["score"] = table["score"].astype(float)
    return table


def read_edge_ranking_table(path: str) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    expected_columns = {"gene_1", "gene_2", "score"}
    missing_columns = expected_columns.difference(set(table.columns))
    if missing_columns:
        raise ValueError(f"Edge table missing columns {sorted(missing_columns)}: {path}")
    table = table[["gene_1", "gene_2", "score"]].copy()
    table["gene_1"] = table["gene_1"].astype(str)
    table["gene_2"] = table["gene_2"].astype(str)
    table["score"] = table["score"].astype(float)
    return table


def build_ranking_series(
    item_score_table: pd.DataFrame,
    item_columns: Sequence[str],
    score_column: str,
) -> pd.Series:
    if len(item_columns) == 1:
        item_identifier = item_score_table[item_columns[0]].astype(str)
    else:
        item_identifier = item_score_table[item_columns].astype(str).agg("|".join, axis=1)
    score_values = item_score_table[score_column].astype(float)

    ranking_table = pd.DataFrame({"item": item_identifier, "score": score_values})
    ranking_table = ranking_table.sort_values(
        by=["score", "item"], ascending=[False, True], kind="mergesort"
    )
    ranking_table = ranking_table.drop_duplicates(subset=["item"], keep="first")
    ranking_series = pd.Series(
        ranking_table["score"].to_numpy(dtype=float),
        index=ranking_table["item"].to_numpy(dtype=str),
        name="score",
    )
    return ranking_series


def compute_top_k_set(ranking_series: pd.Series, top_k: int) -> set:
    if top_k <= 0:
        return set()
    top_items = ranking_series.sort_values(ascending=False, kind="mergesort").head(top_k).index
    return set(top_items.astype(str).tolist())


def pairwise_fold_pairs(number_of_folds: int) -> List[Tuple[int, int]]:
    return [(first, second) for first in range(number_of_folds) for second in range(first + 1, number_of_folds)]


def kuncheva_index(intersection_size: int, top_k: int, universe_size: int) -> float:
    """Kuncheva index for feature set stability.

    Let r = |A ∩ B|, |A| = |B| = k, and total universe size N.
    Kuncheva(A,B) = (r - k^2/N) / (k - k^2/N)
    """
    if universe_size <= 0:
        return np.nan
    expected_intersection = (top_k * top_k) / float(universe_size)
    denominator = top_k - expected_intersection
    if denominator == 0:
        return np.nan
    return float((intersection_size - expected_intersection) / denominator)


def compute_rank_vectors(
    ranking_series_by_fold: Dict[int, pd.Series],
    universe_items: List[str],
) -> Dict[int, np.ndarray]:
    """Convert scores to ranks per fold on a fixed universe.

    Rank definition: rank 1 is best (highest score). Missing items get worst rank = len(universe)+1.
    """
    universe_size = len(universe_items)
    worst_rank = universe_size + 1

    rank_vectors_by_fold: Dict[int, np.ndarray] = {}

    for fold_index, ranking_series in ranking_series_by_fold.items():
        sorted_items = ranking_series.sort_values(ascending=False, kind="mergesort")
        rank_index = pd.Series(
            np.arange(1, len(sorted_items) + 1, dtype=np.int64),
            index=sorted_items.index.astype(str),
            name="rank",
        )
        ranks = np.full((universe_size,), worst_rank, dtype=np.int64)
        for position, item in enumerate(universe_items):
            if item in rank_index.index:
                ranks[position] = int(rank_index.loc[item])
        rank_vectors_by_fold[fold_index] = ranks

    return rank_vectors_by_fold


def compute_pairwise_similarity_matrices(
    rank_vectors_by_fold: Dict[int, np.ndarray],
    number_of_folds: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    spearman_matrix = np.full((number_of_folds, number_of_folds), np.nan, dtype=float)
    kendall_matrix = np.full((number_of_folds, number_of_folds), np.nan, dtype=float)

    for fold_index in range(number_of_folds):
        spearman_matrix[fold_index, fold_index] = 1.0
        kendall_matrix[fold_index, fold_index] = 1.0

    for first_fold, second_fold in pairwise_fold_pairs(number_of_folds):
        first_ranks = rank_vectors_by_fold[first_fold]
        second_ranks = rank_vectors_by_fold[second_fold]
        spearman_value = stats.spearmanr(first_ranks, second_ranks).correlation
        kendall_value = stats.kendalltau(first_ranks, second_ranks).correlation
        spearman_matrix[first_fold, second_fold] = float(spearman_value)
        spearman_matrix[second_fold, first_fold] = float(spearman_value)
        kendall_matrix[first_fold, second_fold] = float(kendall_value)
        kendall_matrix[second_fold, first_fold] = float(kendall_value)

    fold_names = [f"fold_{index}" for index in range(number_of_folds)]
    spearman_table = pd.DataFrame(spearman_matrix, index=fold_names, columns=fold_names)
    kendall_table = pd.DataFrame(kendall_matrix, index=fold_names, columns=fold_names)
    return spearman_table, kendall_table


def save_matrix_heatmap(table: pd.DataFrame, output_path: str, title: str) -> None:
    figure = plt.figure(figsize=(7, 6))
    plt.imshow(table.to_numpy(dtype=float), aspect="auto")
    plt.title(title)
    plt.xticks(range(table.shape[1]), table.columns, rotation=90)
    plt.yticks(range(table.shape[0]), table.index)
    plt.colorbar()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(figure)


def compute_top_k_overlap_statistics(
    ranking_series_by_fold: Dict[int, pd.Series],
    number_of_folds: int,
    top_k_values: Sequence[int],
    universe_mode: str,
) -> pd.DataFrame:
    if universe_mode not in {"union", "intersection"}:
        raise ValueError("universe_mode must be 'union' or 'intersection'")

    fold_pairs = pairwise_fold_pairs(number_of_folds)

    if universe_mode == "union":
        universe_set = set()
        for ranking_series in ranking_series_by_fold.values():
            universe_set.update(ranking_series.index.astype(str).tolist())
    else:
        all_fold_item_sets = [set(series.index.astype(str).tolist()) for series in ranking_series_by_fold.values()]
        universe_set = set.intersection(*all_fold_item_sets) if all_fold_item_sets else set()

    universe_size = int(len(universe_set))

    rows: List[Dict[str, float]] = []
    for top_k in top_k_values:
        jaccard_values = []
        kuncheva_values = []
        for first_fold, second_fold in fold_pairs:
            first_set = compute_top_k_set(ranking_series_by_fold[first_fold], top_k)
            second_set = compute_top_k_set(ranking_series_by_fold[second_fold], top_k)
            intersection_size = len(first_set.intersection(second_set))
            union_size = len(first_set.union(second_set))
            jaccard_value = float(intersection_size / union_size) if union_size > 0 else np.nan
            jaccard_values.append(jaccard_value)
            kuncheva_values.append(kuncheva_index(intersection_size, top_k, universe_size))

        rows.append(
            {
                "top_k": int(top_k),
                "universe_size": int(universe_size),
                "jaccard_mean": float(np.nanmean(jaccard_values)) if len(jaccard_values) else np.nan,
                "jaccard_std": float(np.nanstd(jaccard_values)) if len(jaccard_values) else np.nan,
                "kuncheva_mean": float(np.nanmean(kuncheva_values)) if len(kuncheva_values) else np.nan,
                "kuncheva_std": float(np.nanstd(kuncheva_values)) if len(kuncheva_values) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def evaluate_rank_stability(
    ranking_series_by_fold: Dict[int, pd.Series],
    number_of_folds: int,
    universe_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if universe_mode not in {"union", "intersection"}:
        raise ValueError("universe_mode must be 'union' or 'intersection'")

    if universe_mode == "union":
        universe_set = set()
        for ranking_series in ranking_series_by_fold.values():
            universe_set.update(ranking_series.index.astype(str).tolist())
    else:
        all_fold_item_sets = [set(series.index.astype(str).tolist()) for series in ranking_series_by_fold.values()]
        universe_set = set.intersection(*all_fold_item_sets) if all_fold_item_sets else set()

    universe_items = sorted(list(universe_set))
    rank_vectors_by_fold = compute_rank_vectors(ranking_series_by_fold, universe_items)
    spearman_table, kendall_table = compute_pairwise_similarity_matrices(rank_vectors_by_fold, number_of_folds)
    return spearman_table, kendall_table, universe_items


def find_legacy_fold_directories(experiment_root: str, seed: int) -> Dict[int, str]:
    root_path = Path(experiment_root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"experiment_root is not a directory: {experiment_root}")

    found: Dict[int, str] = {}
    pattern = re.compile(rf"^fold_(\d+)_{int(seed)}$")
    for entry in root_path.iterdir():
        if not entry.is_dir():
            continue
        match = pattern.match(entry.name)
        if match is None:
            continue
        fold_index = int(match.group(1))
        found[fold_index] = str(entry)
    return found


def find_split_layout_directories(experiment_root: str, seed: int) -> Dict[int, str]:
    root_path = Path(experiment_root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"experiment_root is not a directory: {experiment_root}")

    found: Dict[int, str] = {}
    pattern = re.compile(r"^split_idx_(\d+)$")
    for entry in root_path.rglob("split_idx_*"):
        if not entry.is_dir():
            continue
        match = pattern.match(entry.name)
        if match is None:
            continue
        fold_index = int(match.group(1))
        seed_directory = entry / f"seed_{int(seed)}"
        if seed_directory.is_dir():
            existing = found.get(fold_index, None)
            candidate = str(seed_directory)
            if existing is None or len(candidate) < len(existing):
                found[fold_index] = candidate
    return found


def choose_layout(
    legacy_directories: Dict[int, str],
    split_directories: Dict[int, str],
    layout: str,
) -> Tuple[str, Dict[int, str]]:
    if layout == "legacy":
        if not legacy_directories:
            raise FileNotFoundError("No legacy fold directories found (fold_{fold}_{seed}).")
        return "legacy", legacy_directories
    if layout == "split":
        if not split_directories:
            raise FileNotFoundError("No split layout directories found (split_idx_{fold}/seed_{seed}).")
        return "split", split_directories

    if legacy_directories and split_directories:
        if len(legacy_directories) >= len(split_directories):
            return "legacy", legacy_directories
        return "split", split_directories
    if legacy_directories:
        return "legacy", legacy_directories
    if split_directories:
        return "split", split_directories
    raise FileNotFoundError(
        "No fold directories found. Checked legacy (fold_{fold}_{seed}) and split "
        "(split_idx_{fold}/seed_{seed}) layouts."
    )


def normalize_fold_directories(
    discovered_directories: Dict[int, str],
    number_of_folds: Optional[int],
) -> Dict[int, str]:
    if number_of_folds is not None:
        fold_directories: Dict[int, str] = {}
        for fold_index in range(int(number_of_folds)):
            if fold_index not in discovered_directories:
                raise FileNotFoundError(
                    f"Missing fold index {fold_index}. Found: {sorted(discovered_directories.keys())}"
                )
            fold_directories[fold_index] = discovered_directories[fold_index]
        return fold_directories

    discovered_indices = sorted(discovered_directories.keys())
    expected_indices = list(range(len(discovered_indices)))
    if discovered_indices != expected_indices:
        raise ValueError(
            "Detected non-consecutive fold indices. "
            f"Found: {discovered_indices}. Either complete missing folds or pass --number_of_folds explicitly."
        )
    return {index: discovered_directories[index] for index in discovered_indices}


def load_fold_directories(
    experiment_root: str,
    seed: int,
    number_of_folds: Optional[int],
    layout: str,
) -> Tuple[str, Dict[int, str]]:
    legacy_directories = find_legacy_fold_directories(experiment_root, seed)
    split_directories = find_split_layout_directories(experiment_root, seed)
    selected_layout, discovered_directories = choose_layout(
        legacy_directories=legacy_directories,
        split_directories=split_directories,
        layout=layout,
    )
    fold_directories = normalize_fold_directories(discovered_directories, number_of_folds)
    return selected_layout, fold_directories


def load_gene_rankings_across_folds(fold_directories: Dict[int, str]) -> Dict[str, Dict[int, pd.Series]]:
    """Returns dict[group_name -> dict[fold -> ranking_series]].

    For this refactor, gene biomarker is overall only.
    """
    group_name = "overall"
    ranking_by_fold: Dict[int, pd.Series] = {}

    for fold_index, fold_directory in fold_directories.items():
        current_path = os.path.join(fold_directory, "biomarker", "final_biomarker.tsv")
        legacy_path = os.path.join(
            fold_directory, "biomarker", "gene_biomarker", f"gene_importance_{fold_index}.tsv"
        )

        if os.path.exists(current_path):
            gene_score_table = pd.read_csv(current_path, sep="\t")
            if "gene" not in gene_score_table.columns or "S_final" not in gene_score_table.columns:
                raise ValueError(f"Current biomarker file missing columns 'gene'/'S_final': {current_path}")
            ranking_series = build_ranking_series(
                gene_score_table.rename(columns={"gene": "gene_symbol", "S_final": "score"}),
                ["gene_symbol"],
                "score",
            )
            ranking_by_fold[fold_index] = ranking_series
            continue

        if os.path.exists(legacy_path):
            gene_score_table = read_gene_ranking_table(legacy_path)
            ranking_series = build_ranking_series(gene_score_table, ["gene_symbol"], "score")
            ranking_by_fold[fold_index] = ranking_series
            continue

        raise FileNotFoundError(
            "Missing gene biomarker file for fold "
            f"{fold_index}. Tried: {current_path}, {legacy_path}"
        )

    return {group_name: ranking_by_fold}


def load_edge_rankings_across_folds(fold_directories: Dict[int, str]) -> Dict[str, Dict[int, pd.Series]]:
    """Returns dict[group_name -> dict[fold -> ranking_series]].

    Groups are inferred from filenames in network_biomarker directory.
    """
    group_rankings: Dict[str, Dict[int, pd.Series]] = {}

    any_edge_found = False
    for fold_index, fold_directory in fold_directories.items():
        network_directory = os.path.join(fold_directory, "biomarker", "network_biomarker")
        if not os.path.isdir(network_directory):
            continue

        edge_files = sorted(Path(network_directory).glob(f"network_edges_*_fold{fold_index}.tsv"))
        if not edge_files:
            continue
        any_edge_found = True

        for edge_file in edge_files:
            file_name = edge_file.name
            group_part = file_name[len("network_edges_") : -len(f"_fold{fold_index}.tsv")]
            group_name = group_part

            edge_score_table = read_edge_ranking_table(str(edge_file))
            ranking_series = build_ranking_series(edge_score_table, ["gene_1", "gene_2"], "score")

            if group_name not in group_rankings:
                group_rankings[group_name] = {}
            group_rankings[group_name][fold_index] = ranking_series

    if not any_edge_found:
        return {}

    # Sanity: ensure all groups have all folds
    for group_name, ranking_by_fold in list(group_rankings.items()):
        missing_folds = [fold for fold in fold_directories.keys() if fold not in ranking_by_fold]
        if missing_folds:
            # Drop incomplete groups to keep evaluation consistent
            del group_rankings[group_name]

    return group_rankings


def evaluate_and_write_stability(
    item_type_name: str,
    group_name: str,
    ranking_series_by_fold: Dict[int, pd.Series],
    number_of_folds: int,
    top_k_values: Sequence[int],
    universe_mode_overlap: str,
    universe_mode_rank: str,
    output_directory: str,
) -> None:
    os.makedirs(output_directory, exist_ok=True)

    overlap_table = compute_top_k_overlap_statistics(
        ranking_series_by_fold,
        number_of_folds=number_of_folds,
        top_k_values=top_k_values,
        universe_mode=universe_mode_overlap,
    )

    overlap_output_csv = os.path.join(
        output_directory, f"{item_type_name}__{safe_string_for_filename(group_name)}__topk_overlap.csv"
    )
    overlap_table.to_csv(overlap_output_csv, index=False)

    # Plot Jaccard / Kuncheva curves
    figure = plt.figure(figsize=(7, 5))
    plt.plot(overlap_table["top_k"].to_numpy(), overlap_table["jaccard_mean"].to_numpy(), label="Jaccard mean")
    plt.plot(overlap_table["top_k"].to_numpy(), overlap_table["kuncheva_mean"].to_numpy(), label="Kuncheva mean")
    plt.xlabel("top-k")
    plt.ylabel("stability")
    plt.title(f"{item_type_name} stability: {group_name}")
    plt.legend()
    plt.tight_layout()
    plot_path_png = os.path.join(
        output_directory, f"{item_type_name}__{safe_string_for_filename(group_name)}__topk_overlap.png"
    )
    plot_path_pdf = os.path.join(
        output_directory, f"{item_type_name}__{safe_string_for_filename(group_name)}__topk_overlap.pdf"
    )
    plt.savefig(plot_path_png, dpi=200)
    plt.savefig(plot_path_pdf)
    plt.close(figure)

    spearman_table, kendall_table, universe_items = evaluate_rank_stability(
        ranking_series_by_fold,
        number_of_folds=number_of_folds,
        universe_mode=universe_mode_rank,
    )

    spearman_output_csv = os.path.join(
        output_directory, f"{item_type_name}__{safe_string_for_filename(group_name)}__spearman.csv"
    )
    kendall_output_csv = os.path.join(
        output_directory, f"{item_type_name}__{safe_string_for_filename(group_name)}__kendall.csv"
    )
    spearman_table.to_csv(spearman_output_csv)
    kendall_table.to_csv(kendall_output_csv)

    save_matrix_heatmap(
        spearman_table,
        os.path.join(output_directory, f"{item_type_name}__{safe_string_for_filename(group_name)}__spearman.png"),
        f"Spearman rank correlation: {item_type_name} / {group_name}",
    )
    save_matrix_heatmap(
        kendall_table,
        os.path.join(output_directory, f"{item_type_name}__{safe_string_for_filename(group_name)}__kendall.png"),
        f"Kendall tau: {item_type_name} / {group_name}",
    )

    # Save universe for transparency
    universe_output_path = os.path.join(
        output_directory, f"{item_type_name}__{safe_string_for_filename(group_name)}__universe_{universe_mode_rank}.txt"
    )
    with open(universe_output_path, "w") as output_file:
        for item in universe_items:
            output_file.write(f"{item}\n")


def load_deg_table(
    deg_table_path: str,
    gene_column: str,
    adjusted_p_value_column: str,
    log2_fold_change_column: str,
) -> pd.DataFrame:
    deg_table = pd.read_csv(deg_table_path, sep=None, engine="python")
    required_columns = {gene_column, adjusted_p_value_column, log2_fold_change_column}
    missing_columns = required_columns.difference(set(deg_table.columns))
    if missing_columns:
        raise ValueError(f"DEG table missing columns {sorted(missing_columns)}")

    deg_table = deg_table[[gene_column, adjusted_p_value_column, log2_fold_change_column]].copy()
    deg_table = deg_table.rename(
        columns={
            gene_column: "gene_symbol",
            adjusted_p_value_column: "adjusted_p_value",
            log2_fold_change_column: "log2_fold_change",
        }
    )
    deg_table["gene_symbol"] = deg_table["gene_symbol"].astype(str)
    deg_table["adjusted_p_value"] = pd.to_numeric(deg_table["adjusted_p_value"], errors="coerce")
    deg_table["log2_fold_change"] = pd.to_numeric(deg_table["log2_fold_change"], errors="coerce")
    deg_table = deg_table.dropna(subset=["adjusted_p_value", "log2_fold_change"])
    return deg_table


def build_deg_set(
    deg_table: pd.DataFrame,
    adjusted_p_value_threshold: float,
    absolute_log2_fold_change_threshold: float,
    direction: str,
) -> set:
    if direction not in {"all", "up", "down"}:
        raise ValueError("direction must be one of: all, up, down")

    filtered = deg_table[
        (deg_table["adjusted_p_value"] <= adjusted_p_value_threshold)
        & (deg_table["log2_fold_change"].abs() >= absolute_log2_fold_change_threshold)
    ].copy()

    if direction == "up":
        filtered = filtered[filtered["log2_fold_change"] >= absolute_log2_fold_change_threshold]
    elif direction == "down":
        filtered = filtered[filtered["log2_fold_change"] <= -absolute_log2_fold_change_threshold]

    return set(filtered["gene_symbol"].astype(str).tolist())


def hypergeometric_enrichment_p_value(
    universe_size: int,
    deg_set_size: int,
    top_k: int,
    overlap_count: int,
) -> float:
    if universe_size <= 0:
        return np.nan
    if deg_set_size <= 0:
        return 1.0
    if top_k <= 0:
        return 1.0
    # P(X >= x)
    survival_value = stats.hypergeom.sf(overlap_count - 1, universe_size, deg_set_size, top_k)
    return float(survival_value)


def evaluate_deg_agreement(
    gene_ranking_by_fold: Dict[int, pd.Series],
    number_of_folds: int,
    top_k_values: Sequence[int],
    universe_items: List[str],
    deg_set: set,
    output_directory: str,
) -> None:
    os.makedirs(output_directory, exist_ok=True)

    universe_set = set(universe_items)
    deg_set_in_universe = set([g for g in deg_set if g in universe_set])

    rows: List[Dict[str, float]] = []
    for fold_index in range(number_of_folds):
        ranking_series = gene_ranking_by_fold[fold_index]
        for top_k in top_k_values:
            top_set = compute_top_k_set(ranking_series, top_k)
            overlap_count = len(top_set.intersection(deg_set_in_universe))
            overlap_fraction = float(overlap_count / float(top_k)) if top_k > 0 else np.nan
            p_value = hypergeometric_enrichment_p_value(
                universe_size=len(universe_items),
                deg_set_size=len(deg_set_in_universe),
                top_k=top_k,
                overlap_count=overlap_count,
            )
            rows.append(
                {
                    "fold": int(fold_index),
                    "top_k": int(top_k),
                    "universe_size": int(len(universe_items)),
                    "deg_set_size": int(len(deg_set_in_universe)),
                    "overlap_count": int(overlap_count),
                    "overlap_fraction": float(overlap_fraction),
                    "hypergeom_p_value": float(p_value),
                }
            )

    result_table = pd.DataFrame(rows)
    if len(result_table) > 0:
        p_values = result_table["hypergeom_p_value"].to_numpy(dtype=float)
        adjusted = multipletests(p_values, alpha=0.05, method="fdr_bh")
        result_table["hypergeom_fdr_bh"] = adjusted[1]

    result_table.to_csv(os.path.join(output_directory, "deg_overlap_hypergeom.csv"), index=False)

    # Aggregate by top_k
    aggregated = (
        result_table.groupby("top_k", as_index=False)
        .agg(
            overlap_fraction_mean=("overlap_fraction", "mean"),
            overlap_fraction_std=("overlap_fraction", "std"),
            hypergeom_p_value_median=("hypergeom_p_value", "median"),
            hypergeom_fdr_bh_median=("hypergeom_fdr_bh", "median"),
        )
        .sort_values("top_k")
    )
    aggregated.to_csv(os.path.join(output_directory, "deg_overlap_hypergeom_aggregated.csv"), index=False)

    figure = plt.figure(figsize=(7, 5))
    plt.errorbar(
        aggregated["top_k"].to_numpy(),
        aggregated["overlap_fraction_mean"].to_numpy(),
        yerr=aggregated["overlap_fraction_std"].fillna(0).to_numpy(),
        fmt="-o",
    )
    plt.xlabel("top-k")
    plt.ylabel("overlap fraction")
    plt.title("DEG overlap with model top-k genes")
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "deg_overlap_fraction.png"), dpi=200)
    plt.savefig(os.path.join(output_directory, "deg_overlap_fraction.pdf"))
    plt.close(figure)


def compute_deg_signed_statistic(deg_table: pd.DataFrame) -> pd.Series:
    statistic = -np.log10(deg_table["adjusted_p_value"].clip(lower=1e-300))
    sign = np.sign(deg_table["log2_fold_change"].to_numpy(dtype=float))
    signed_statistic = statistic.to_numpy(dtype=float) * sign
    return pd.Series(signed_statistic, index=deg_table["gene_symbol"].astype(str), name="deg_signed_stat")


def gsea_like_enrichment_curve(
    ranking_series: pd.Series,
    deg_set: set,
    universe_items: List[str],
    output_path_prefix: str,
) -> None:
    universe_set = set(universe_items)
    deg_set_in_universe = set([g for g in deg_set if g in universe_set])

    ordered_items = ranking_series.sort_values(ascending=False, kind="mergesort").index.astype(str).tolist()
    ordered_items = [g for g in ordered_items if g in universe_set]

    if not ordered_items:
        return

    hit_indicator = np.array([1 if g in deg_set_in_universe else 0 for g in ordered_items], dtype=float)
    number_of_hits = float(hit_indicator.sum())
    number_of_misses = float(len(hit_indicator) - hit_indicator.sum())
    if number_of_hits == 0 or number_of_misses == 0:
        return

    hit_step = 1.0 / number_of_hits
    miss_step = 1.0 / number_of_misses

    running_sum = []
    current_sum = 0.0
    for is_hit in hit_indicator:
        current_sum += hit_step if is_hit == 1 else -miss_step
        running_sum.append(current_sum)

    running_sum = np.asarray(running_sum, dtype=float)
    enrichment_score = float(running_sum.max())

    figure = plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(running_sum)), running_sum)
    plt.axhline(0.0)
    plt.title(f"GSEA-like enrichment curve (ES={enrichment_score:.3f})")
    plt.xlabel("Rank position")
    plt.ylabel("Running sum")
    plt.tight_layout()
    plt.savefig(output_path_prefix + ".png", dpi=200)
    plt.savefig(output_path_prefix + ".pdf")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate biomarker stability across folds and optional DEG agreement."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument(
        "--experiment_root",
        type=str,
        default=None,
        help="Directory that contains fold outputs (e.g., ./experiment/aorta).",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--number_of_folds", type=int, default=None)
    parser.add_argument(
        "--layout",
        choices=["auto", "legacy", "split"],
        default="auto",
        help="legacy: fold_{fold}_{seed}, split: split_idx_{fold}/seed_{seed}",
    )
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--top_k_values", type=int, nargs="+", default=[20, 50, 100, 200])
    parser.add_argument("--universe_mode_overlap", choices=["union", "intersection"], default="union")
    parser.add_argument("--universe_mode_rank", choices=["union", "intersection"], default="intersection")

    parser.add_argument("--deg_table", type=str, default=None)
    parser.add_argument("--deg_gene_column", type=str, default="gene")
    parser.add_argument("--deg_padj_column", type=str, default="adjusted_p_value")
    parser.add_argument("--deg_log2fc_column", type=str, default="log2_fold_change")
    parser.add_argument("--deg_padj_threshold", type=float, default=0.05)
    parser.add_argument("--deg_abs_log2fc_threshold", type=float, default=0.25)
    parser.add_argument("--deg_direction", choices=["all", "up", "down"], default="all")

    arguments = parser.parse_args()

    config_dictionary: Dict = {}
    if arguments.config is not None:
        config_dictionary = load_config_dictionary(arguments.config)

    dataset_name: Optional[str] = arguments.dataset
    if dataset_name is None and "dataset" in config_dictionary:
        dataset_name = str(config_dictionary["dataset"])

    experiment_root: Optional[str] = arguments.experiment_root
    if experiment_root is None and "experiment_root" in config_dictionary:
        base_experiment_root = str(config_dictionary["experiment_root"])
        if dataset_name:
            dataset_root = os.path.join(base_experiment_root, dataset_name)
            run_signature = "_".join(
                [
                    str(bool(config_dictionary.get("prototype_prior_enabled", False))),
                    str(config_dictionary.get("protein_embedding_merge", "mean")),
                    str(config_dictionary.get("sample_bags", "proportional")),
                    str(config_dictionary.get("pooling", "flat_attention")),
                ]
            )
            signature_root = os.path.join(dataset_root, run_signature)
            if os.path.isdir(signature_root):
                experiment_root = signature_root
            else:
                experiment_root = dataset_root
        else:
            experiment_root = base_experiment_root

    if experiment_root is None:
        raise ValueError("Provide --experiment_root, or provide --config containing `experiment_root`.")

    if arguments.output_root is None:
        output_name = f"{dataset_name}_seed{int(arguments.seed)}" if dataset_name else f"seed{int(arguments.seed)}"
        evaluation_root = os.path.join("./biomarker_evaluation", output_name)
    else:
        evaluation_root = arguments.output_root
    os.makedirs(evaluation_root, exist_ok=True)

    selected_layout, fold_directories = load_fold_directories(
        experiment_root=experiment_root,
        seed=int(arguments.seed),
        number_of_folds=arguments.number_of_folds,
        layout=str(arguments.layout),
    )
    number_of_folds = len(fold_directories)

    print(f"[0/4] Layout: {selected_layout}, folds: {sorted(fold_directories.keys())}")
    print("[1/4] Loading gene biomarker rankings across folds")
    gene_group_rankings = load_gene_rankings_across_folds(fold_directories)

    print("[2/4] Loading network biomarker rankings across folds")
    edge_group_rankings = load_edge_rankings_across_folds(fold_directories)

    print("[3/4] Evaluating cross-fold stability")
    for group_name, ranking_by_fold in gene_group_rankings.items():
        output_directory = os.path.join(evaluation_root, "stability", "genes")
        evaluate_and_write_stability(
            item_type_name="genes",
            group_name=group_name,
            ranking_series_by_fold=ranking_by_fold,
            number_of_folds=number_of_folds,
            top_k_values=arguments.top_k_values,
            universe_mode_overlap=arguments.universe_mode_overlap,
            universe_mode_rank=arguments.universe_mode_rank,
            output_directory=output_directory,
        )

    if len(edge_group_rankings) == 0:
        print("No network edge biomarker files found; skipping edge stability evaluation.")
    else:
        for group_name, ranking_by_fold in edge_group_rankings.items():
            output_directory = os.path.join(evaluation_root, "stability", "edges")
            evaluate_and_write_stability(
                item_type_name="edges",
                group_name=group_name,
                ranking_series_by_fold=ranking_by_fold,
                number_of_folds=number_of_folds,
                top_k_values=arguments.top_k_values,
                universe_mode_overlap=arguments.universe_mode_overlap,
                universe_mode_rank=arguments.universe_mode_rank,
                output_directory=output_directory,
            )

    # DEG evaluation (genes only)
    if arguments.deg_table is not None:
        print("[4/4] DEG agreement evaluation")
        deg_table = load_deg_table(
            deg_table_path=arguments.deg_table,
            gene_column=arguments.deg_gene_column,
            adjusted_p_value_column=arguments.deg_padj_column,
            log2_fold_change_column=arguments.deg_log2fc_column,
        )
        deg_set = build_deg_set(
            deg_table,
            adjusted_p_value_threshold=float(arguments.deg_padj_threshold),
            absolute_log2_fold_change_threshold=float(arguments.deg_abs_log2fc_threshold),
            direction=str(arguments.deg_direction),
        )

        gene_group_name = "overall"
        if gene_group_name not in gene_group_rankings:
            available_groups = sorted(gene_group_rankings.keys())
            raise ValueError(
                f"Expected gene group '{gene_group_name}' not found. Available groups: {available_groups}"
            )

        # Universe: use rank-universe mode and reuse as DEG background.
        gene_ranking_by_fold = gene_group_rankings[gene_group_name]
        _, _, universe_items = evaluate_rank_stability(
            gene_ranking_by_fold,
            number_of_folds=number_of_folds,
            universe_mode=arguments.universe_mode_rank,
        )

        deg_output_directory = os.path.join(evaluation_root, "deg_agreement")
        evaluate_deg_agreement(
            gene_ranking_by_fold=gene_ranking_by_fold,
            number_of_folds=number_of_folds,
            top_k_values=arguments.top_k_values,
            universe_items=universe_items,
            deg_set=deg_set,
            output_directory=deg_output_directory,
        )

        # Threshold-free plot (GSEA-like) using fold-averaged ranking
        mean_score_table = pd.DataFrame({
            fold_index: series for fold_index, series in gene_ranking_by_fold.items()
        }).fillna(0.0)
        mean_score_series = mean_score_table.mean(axis=1)
        gsea_like_enrichment_curve(
            ranking_series=mean_score_series,
            deg_set=deg_set,
            universe_items=universe_items,
            output_path_prefix=os.path.join(deg_output_directory, "gsea_like_enrichment_mean_ranking"),
        )

    print("Done. Outputs under:", os.path.abspath(evaluation_root))
