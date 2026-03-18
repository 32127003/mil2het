from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scanpy as sc

from scbiomarker import preselection

from smoke_test_helpers import assert_module_endpoints, print_success

PRESELECTION_ENDPOINTS = [
    "RWRConfig",
    "read_gene_list",
    "write_gene_list",
    "deduplicate_preserve_order",
    "load_ppi_node_set",
    "remap_var_names_to_best_symbol_source",
    "load_split_train_indices",
    "collect_deg_rows",
    "select_deg",
    "select_deg_union_by_celltype",
    "compute_global_deg_zscore",
    "run_rwr",
    "np_scores",
    "infer_dataset_name",
    "resolve_adata_path",
    "resolve_preselection_output_root",
    "discover_split_indices",
    "preselection",
    "run_split_preselection",
    "build_legacy_preselection_config",
    "build_preselection_config_from_cli_args",
]


def build_deg_adata() -> sc.AnnData:
    num_cells = 40
    num_genes = 4
    x = np.random.randn(num_cells, num_genes).astype(np.float32)

    labels = np.array(["1"] * 20 + ["0"] * 20, dtype=object)
    x[:20, 0] += 2.0
    x[20:, 1] += 1.5

    celltypes = np.array(["T", "B"] * 20, dtype=object)

    adata = sc.AnnData(X=x)
    adata.var_names = pd.Index(["G1", "G2", "G3", "G4"])
    adata.obs["label"] = labels
    adata.obs["celltype"] = celltypes
    return adata


def test_preselection_endpoints_exist() -> None:
    assert_module_endpoints(
        preselection,
        PRESELECTION_ENDPOINTS,
        module_label="scbiomarker.preselection",
    )


def test_preselection_io_and_deg_helpers() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        genes_path = temp_path / "genes.tsv"
        preselection.write_gene_list(str(genes_path), genes=["G1", "G2"], scores=[0.9, 0.7])
        loaded_genes = preselection.read_gene_list(str(genes_path))
        assert loaded_genes == ["G1", "G2"]

        deduped = preselection.deduplicate_preserve_order(["G1", "G2", "G1", "G3"])
        assert deduped == ["G1", "G2", "G3"]

        ppi_path = temp_path / "ppi.tsv"
        ppi_path.write_text("G1\tG2\nG2\tG3\nG3\tG4\n", encoding="utf-8")
        ppi_nodes = preselection.load_ppi_node_set(str(ppi_path))
        assert ppi_nodes == {"G1", "G2", "G3", "G4"}

        split_path = temp_path / "toy_idx_0.pkl"
        with split_path.open("wb") as file_handle:
            pickle.dump([[0, 1, 2], [3], [4]], file_handle)
        train_indices = preselection.load_split_train_indices(str(temp_path), "toy", 0)
        assert train_indices.tolist() == [0, 1, 2]

        adata_remap = sc.AnnData(X=np.random.randn(8, 4).astype(np.float32))
        adata_remap.var_names = pd.Index(["ENSG1", "ENSG2", "ENSG3", "ENSG4"])
        adata_remap.var["gene_symbol"] = ["G1", "G2", "G3", "G4"]
        remapped_adata, remap_info = preselection.remap_var_names_to_best_symbol_source(
            adata=adata_remap,
            ppi_node_set=ppi_nodes,
        )
        assert remap_info["source"] == "gene_symbol"
        assert remapped_adata.var_names.tolist() == ["G1", "G2", "G3", "G4"]

        adata_deg = build_deg_adata()
        deg_rows = preselection.collect_deg_rows(
            adata=adata_deg,
            groupby="label",
            groups=["1", "0"],
            ppi_node_set=ppi_nodes,
            max_p_value=1.0,
            min_abs_logfc=0.0,
            apply_thresholds=True,
        )
        assert not deg_rows.empty

        pass_genes, pass_pvals, pass_logfc = preselection.select_deg(
            adata=adata_deg,
            groupby="label",
            groups=["1", "0"],
            ppi_node_set=ppi_nodes,
            max_p_value=1.0,
            min_abs_logfc=0.0,
        )
        assert len(pass_genes) == len(pass_pvals) == len(pass_logfc)
        assert len(pass_genes) > 0

        (
            union_genes,
            _,
            _,
            summary,
            celltype_counts_df,
            overlap_counts_df,
            gene_celltype_map,
            _,
        ) = preselection.select_deg_union_by_celltype(
            adata=adata_deg,
            celltype_column="celltype",
            label_column="label",
            label_groups=["1", "0"],
            ppi_node_set=ppi_nodes,
            max_p_value=1.0,
            min_abs_logfc=0.0,
            allow_empty_pass_set=False,
        )
        assert len(union_genes) > 0
        assert summary["celltypes_total"] >= 2
        assert not celltype_counts_df.empty
        assert not overlap_counts_df.empty
        assert len(gene_celltype_map) > 0

        zscore_df = preselection.compute_global_deg_zscore(
            adata=adata_deg,
            groupby="label",
            positive_label="1",
            negative_label="0",
            ppi_node_set=ppi_nodes,
            method="t-test",
        )
        assert "zscore" in zscore_df.columns


def test_preselection_rwr_and_pipeline() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        ppi_path = temp_path / "ppi.tsv"
        ppi_path.write_text("G1\tG2\nG2\tG3\nG3\tG4\n", encoding="utf-8")

        adata = build_deg_adata()

        rwr_cfg = preselection.RWRConfig(
            restart_probability=0.1,
            convergence_threshold_l1=1e-8,
            max_iterations=200,
            treat_as_undirected=True,
        )

        genes_sorted, scores_sorted = preselection.np_scores(
            adata=adata,
            network_path=str(ppi_path),
            seed_genes=["G1", "G2"],
            rwr_config=rwr_cfg,
        )
        assert len(genes_sorted) == len(scores_sorted)
        assert len(genes_sorted) > 0

        node_order, edges = preselection.parse_ppi_edges_and_nodes_in_adata_namespace(
            edge_list_path=str(ppi_path),
            allowed_gene_set=set(adata.var_names.tolist()),
        )
        node_to_index = {gene: i for i, gene in enumerate(node_order)}
        p_matrix = preselection.build_row_normalized_transposed_adjacency(
            node_order=node_order,
            edges=edges,
            treat_as_undirected=True,
        )
        p0 = preselection.seed_vector(["G1", "G2"], node_to_index=node_to_index)
        rwr_scores = preselection.run_rwr(p_matrix=p_matrix, p0=p0, rwr_config=rwr_cfg)
        assert rwr_scores.shape[0] == len(node_order)

        out_dir = temp_path / "preselection_out"
        preselection.preselection(
            adata=adata,
            ppi_network_path=str(ppi_path),
            groupby="label",
            group1="1",
            group2="0",
            celltype_column="celltype",
            out_dir=str(out_dir),
            split_idx=0,
            deg_max_p_value=1.0,
            deg_min_abs_logfc=0.0,
            restart_prob=0.1,
            convergence_threshold_l1=1e-8,
            max_iterations=200,
            directed=False,
        )

        assert (out_dir / "DEG" / "DEG_merged.tsv").is_file()
        assert (out_dir / "DEG" / "DEG_zscore_global.tsv").is_file()
        assert (out_dir / "NP" / "NP_max.tsv").is_file()


def test_run_split_preselection_with_explicit_inputs() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        ppi_path = temp_path / "ppi.tsv"
        ppi_path.write_text("G1\tG2\nG2\tG3\nG3\tG4\n", encoding="utf-8")

        adata = build_deg_adata()
        patient_ids = [f"p{i // 4}" for i in range(adata.n_obs)]
        adata.obs["patient_id"] = patient_ids
        adata_path = temp_path / "toy_data.h5ad"
        adata.write_h5ad(adata_path)

        splits_dir = temp_path / "splits"
        splits_dir.mkdir(parents=True)
        with (splits_dir / "toy_idx_0.pkl").open("wb") as file_handle:
            pickle.dump(
                [
                    list(range(10)) + list(range(20, 30)),
                    list(range(10, 15)) + list(range(30, 35)),
                    list(range(15, 20)) + list(range(35, 40)),
                ],
                file_handle,
            )

        config = SimpleNamespace(
            adata_path=str(adata_path),
            dataset="",
            label_column="label",
            celltype_column="celltype",
            ppi_path=str(ppi_path),
            splits_directory=str(splits_dir),
            out_dir=str(temp_path / "preselection"),
            num_folds=1,
            binary_positive_label="1",
            binary_negative_label="0",
            deg_max_p_value=1.0,
            deg_min_abs_logfc=0.0,
            restart_prob=0.1,
            convergence_threshold_l1=1e-8,
            max_iterations=200,
            directed=False,
        )

        assert preselection.infer_dataset_name(config) == "toy"
        assert preselection.resolve_adata_path(config) == str(adata_path)
        assert preselection.discover_split_indices(str(splits_dir), "toy", num_folds=1) == [0]

        output_root = preselection.run_split_preselection(config)
        assert output_root == str(temp_path / "preselection")
        assert (Path(output_root) / "split_0" / "DEG" / "DEG_merged.tsv").is_file()
        assert (Path(output_root) / "split_0" / "NP" / "NP_max.tsv").is_file()


def main() -> None:
    test_preselection_endpoints_exist()
    test_preselection_io_and_deg_helpers()
    test_preselection_rwr_and_pipeline()
    test_run_split_preselection_with_explicit_inputs()
    print_success("preselection endpoints")


if __name__ == "__main__":
    main()
