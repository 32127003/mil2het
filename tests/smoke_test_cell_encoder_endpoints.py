from __future__ import annotations

import torch
from mil2het import CellEncoder

from smoke_test_helpers import assert_module_endpoints, print_success

CELL_ENCODER_ENDPOINTS = [
    "GraphAttentionLayer",
    "TransformerConvLayer",
    "GraphCellEncoder",
    "TransformerConvCellEncoder",
    "scatter_softmax",
    "infer_batch_chunk_size",
]


def ring_edge_index(num_nodes: int) -> torch.Tensor:
    src = torch.arange(num_nodes, dtype=torch.long)
    dst = (src + 1) % num_nodes
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)


def build_regulation_onehot(num_celltypes: int, num_nodes: int) -> torch.Tensor:
    regulation = torch.zeros(num_celltypes, num_nodes, 3)
    labels = torch.randint(0, 3, (num_celltypes, num_nodes))
    regulation.scatter_(2, labels.unsqueeze(-1), 1.0)
    return regulation


def test_cell_encoder_endpoints_exist() -> None:
    assert_module_endpoints(
        CellEncoder,
        CELL_ENCODER_ENDPOINTS,
        module_label="mil2het.CellEncoder",
    )


def test_cell_encoder_layers_and_utils() -> None:
    logits = torch.tensor([[[1.0], [2.0], [0.5], [1.5]]], dtype=torch.float32)
    destination_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    attention = CellEncoder.scatter_softmax(logits, destination_index, num_nodes=2)
    assert attention.shape == logits.shape
    assert torch.allclose(attention[0, :2, 0].sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(attention[0, 2:, 0].sum(), torch.tensor(1.0), atol=1e-5)

    chunk_size = CellEncoder.infer_batch_chunk_size(
        batch_size=32,
        num_edges=256,
        feature_dimension=64,
        dtype=torch.float32,
        max_message_memory_mb=0.5,
    )
    assert 1 <= chunk_size <= 32

    node_hidden = torch.randn(2, 8, 16)
    edge_index = ring_edge_index(8)

    gat = CellEncoder.GraphAttentionLayer(
        input_dimension=16,
        output_dimension=16,
        number_of_heads=2,
        dropout_probability=0.0,
        attention_logit_clamp=5.0,
        max_message_memory_mb=64.0,
    )
    gat_out = gat(node_hidden=node_hidden, edge_index=edge_index, return_attention=True)
    assert gat_out["node_hidden"].shape == (2, 8, 16)
    assert gat_out["attention"].shape == (2, edge_index.shape[1], 2)

    transformer = CellEncoder.TransformerConvLayer(
        input_dimension=16,
        output_dimension=16,
        number_of_heads=2,
        dropout_probability=0.0,
        attention_logit_clamp=5.0,
        max_message_memory_mb=64.0,
    )
    transformer_out = transformer(node_hidden=node_hidden, edge_index=edge_index, return_attention=True)
    assert transformer_out["node_hidden"].shape == (2, 8, 16)
    assert transformer_out["attention"].shape == (2, edge_index.shape[1], 2)


def test_graph_and_transformer_cell_encoders_forward() -> None:
    torch.manual_seed(0)

    num_nodes = 16
    num_cells = 6
    num_celltypes = 4
    prior_dim = 32

    edge_index = ring_edge_index(num_nodes)
    protein_prior_embeddings = torch.randn(num_nodes, prior_dim)
    global_zscore = torch.randn(num_nodes)
    regulation_onehot = build_regulation_onehot(num_celltypes=num_celltypes, num_nodes=num_nodes)
    expression_values = torch.randn(num_cells, num_nodes)
    celltype_index = torch.randint(0, num_celltypes, (num_cells,), dtype=torch.long)

    graph_encoder = CellEncoder.GraphCellEncoder(
        num_nodes=num_nodes,
        hidden_dimension=12,
        number_of_layers=2,
        number_of_heads=3,
        dropout_probability=0.1,
        edge_index=edge_index,
        protein_prior_embeddings=protein_prior_embeddings,
        global_zscore=global_zscore,
        regulation_onehot=regulation_onehot,
        num_celltypes=num_celltypes,
        num_treatments=0,
        attention_logit_clamp=5.0,
        expression_feature_scale=1.0,
        max_message_memory_mb=128,
        graph_readout="mean",
        prior_injection_enabled=False,
    )

    graph_encoder.eval()
    with torch.no_grad():
        graph_out = graph_encoder(
            expression_values=expression_values,
            celltype_index=celltype_index,
            return_node_outputs=True,
            return_attention=True,
        )

    assert graph_out["cell_embeddings"].shape == (num_cells, 12)
    assert graph_out["node_scores"].shape == (num_cells, num_nodes)
    assert graph_out["node_hidden"].shape == (num_cells, num_nodes, 12)

    transformer_encoder = CellEncoder.TransformerConvCellEncoder(
        num_nodes=num_nodes,
        hidden_dimension=12,
        number_of_layers=2,
        number_of_heads=3,
        dropout_probability=0.1,
        edge_index=edge_index,
        protein_prior_embeddings=protein_prior_embeddings,
        global_zscore=global_zscore,
        regulation_onehot=regulation_onehot,
        num_celltypes=num_celltypes,
        num_treatments=0,
        attention_logit_clamp=5.0,
        expression_feature_scale=1.0,
        max_message_memory_mb=128,
        graph_readout="mean",
        prior_injection_enabled=False,
    )

    transformer_encoder.eval()
    with torch.no_grad():
        transformer_out = transformer_encoder(
            expression_values=expression_values,
            celltype_index=celltype_index,
            return_node_outputs=True,
            return_attention=True,
        )

    assert transformer_out["cell_embeddings"].shape == (num_cells, 12)
    assert transformer_out["node_scores"].shape == (num_cells, num_nodes)
    assert transformer_out["node_hidden"].shape == (num_cells, num_nodes, 12)


def main() -> None:
    test_cell_encoder_endpoints_exist()
    test_cell_encoder_layers_and_utils()
    test_graph_and_transformer_cell_encoders_forward()
    print_success("CellEncoder endpoints")


if __name__ == "__main__":
    main()
