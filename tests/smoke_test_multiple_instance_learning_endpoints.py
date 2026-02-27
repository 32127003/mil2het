from __future__ import annotations

import torch
from scbiomarker import MultipleInstanceLearning

from smoke_test_helpers import assert_module_endpoints, print_success

MULTIPLE_INSTANCE_LEARNING_ENDPOINTS = [
    "scatter_softmax_1d",
    "GatedAttentionMIL",
    "PatientMILAggregator",
]


def test_mil_endpoints_exist() -> None:
    assert_module_endpoints(
        MultipleInstanceLearning,
        MULTIPLE_INSTANCE_LEARNING_ENDPOINTS,
        module_label="scbiomarker.MultipleInstanceLearning",
    )


def test_mil_endpoints_functionality() -> None:
    logits = torch.tensor([0.0, 1.0, -1.0, 2.0, 0.5, -0.5], dtype=torch.float32)
    segment_index = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    attention = MultipleInstanceLearning.scatter_softmax_1d(logits, segment_index, num_segments=2)
    assert attention.shape == (6,)
    assert torch.allclose(attention[:3].sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(attention[3:].sum(), torch.tensor(1.0), atol=1e-5)

    embeddings = torch.randn(6, 8)
    gated_pool = MultipleInstanceLearning.GatedAttentionMIL(input_dimension=8, hidden_dimension=4)
    gated_out = gated_pool(embeddings=embeddings, segment_index=segment_index, num_segments=2)
    assert gated_out["pooled"].shape == (2, 8)
    assert gated_out["attention"].shape == (6,)

    aggregator = MultipleInstanceLearning.PatientMILAggregator(
        embedding_dimension=8,
        num_celltypes=3,
        pooling="attention",
        attention_hidden_dimension=4,
        recursive_steps=1,
    )
    celltype_index = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    bag_index = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    aggregate_out = aggregator(
        cell_embeddings=embeddings,
        celltype_index=celltype_index,
        bag_index=bag_index,
        num_bags=2,
    )
    assert aggregate_out["patient_embeddings"].shape == (2, 8)
    assert aggregate_out["gamma_attention"].shape == (6,)


def main() -> None:
    test_mil_endpoints_exist()
    test_mil_endpoints_functionality()
    print_success("MultipleInstanceLearning endpoints")


if __name__ == "__main__":
    main()
