from __future__ import annotations

import torch
from mil2het import prior_interface_find

from smoke_test_helpers import assert_module_endpoints, print_success

PRIOR_INTERFACE_FIND_ENDPOINTS = [
    "MultiViewPriorInterfaceFIND",
]


def test_prior_interface_find_endpoint_exists() -> None:
    assert_module_endpoints(
        prior_interface_find,
        PRIOR_INTERFACE_FIND_ENDPOINTS,
        module_label="mil2het.prior_interface_find",
    )


def test_prior_interface_find_forward_paths() -> None:
    num_nodes = 10
    hidden_dimension = 8

    model = prior_interface_find.MultiViewPriorInterfaceFIND(
        frozen_embeddings_by_view={
            "view_a": torch.randn(num_nodes, 6),
            "view_b": torch.randn(num_nodes, 5),
        },
        absolute_query_input_dimension=4,
        hidden_dimension=hidden_dimension,
        number_of_heads=2,
        dropout_probability=0.0,
        feedforward_hidden_dimension=16,
    )

    prompt_tokens = model.build_prompt_tokens()
    assert prompt_tokens.shape == (num_nodes, 2, hidden_dimension)

    node_features = torch.randn(3, num_nodes, 4)
    absolute = model.forward_absolute(node_features=node_features, prompt_tokens=prompt_tokens)
    assert absolute.shape == (3, num_nodes, hidden_dimension)

    relational = model.forward_relational(prompt_tokens=prompt_tokens)
    assert relational.shape == (num_nodes, hidden_dimension)

    edge_index = model.build_sparse_knn_edge_index(k=2, symmetric=True)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] > 0


def main() -> None:
    test_prior_interface_find_endpoint_exists()
    test_prior_interface_find_forward_paths()
    print_success("prior_interface_find endpoints")


if __name__ == "__main__":
    main()
