"""
usage example:
python make_embeddings.py \
  --ppi /data2/project/bin_jip/Biomarker/data/ppi_network.tsv \
  --mapping /data2/project/bin_jip/Biomarker/data/mapping.tsv \
  --fasta /data2/project/bin_jip/Biomarker/data/9606.protein.sequences.v12.0.fa \
  --output ./gene_embedding/ESM3_embeddings.pkl \
  --device cuda:0 \
  --max-len 2048
"""

#!/usr/bin/env python3
import argparse
import csv
import gc
import os
import pickle
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_unique_gene_symbols_from_ppi_network(ppi_network_tsv_path: str) -> List[str]:
    """
    Reads ppi_network.tsv with header columns: protein1, protein2
    Returns a sorted list of unique gene symbols (nodes).
    """
    unique_gene_symbol_set: Set[str] = set()

    with open(ppi_network_tsv_path, "r", newline="") as file_handle:
        reader = csv.DictReader(file_handle, delimiter="\t")
        if reader.fieldnames is None or "protein1" not in reader.fieldnames or "protein2" not in reader.fieldnames:
            raise ValueError("ppi_network.tsv must have header columns: protein1, protein2")

        for row in reader:
            gene_symbol_1 = (row.get("protein1") or "").strip()
            gene_symbol_2 = (row.get("protein2") or "").strip()
            if gene_symbol_1:
                unique_gene_symbol_set.add(gene_symbol_1)
            if gene_symbol_2:
                unique_gene_symbol_set.add(gene_symbol_2)

    return sorted(unique_gene_symbol_set)


def load_string_protein_ids_by_gene(mapping_tsv_path: str) -> Dict[str, Set[str]]:
    """
    Reads mapping.tsv with header columns: string_protein_id, gene
    Returns:
        string_protein_ids_by_gene[gene_symbol] = set({string_protein_id, ...})
    """
    string_protein_ids_by_gene: Dict[str, Set[str]] = {}

    with open(mapping_tsv_path, "r", newline="") as file_handle:
        reader = csv.DictReader(file_handle, delimiter="\t")
        if reader.fieldnames is None or "string_protein_id" not in reader.fieldnames or "gene" not in reader.fieldnames:
            raise ValueError("mapping.tsv must have header columns: string_protein_id, gene")

        for row in reader:
            string_protein_id = (row.get("string_protein_id") or "").strip()
            gene_symbol = (row.get("gene") or "").strip()

            if not string_protein_id or not gene_symbol:
                continue

            if gene_symbol not in string_protein_ids_by_gene:
                string_protein_ids_by_gene[gene_symbol] = set()

            string_protein_ids_by_gene[gene_symbol].add(string_protein_id)

    return string_protein_ids_by_gene


def load_fasta_sequences_by_string_id(fasta_path: str) -> Dict[str, str]:
    """
    Parses a FASTA where headers are like:
        >9606.ENSP00000000233
        MGLTV...
    Returns:
        protein_sequences_by_string_id[string_protein_id] = sequence
    """
    protein_sequences_by_string_id: Dict[str, str] = {}

    current_string_protein_id: Optional[str] = None
    sequence_fragments: List[str] = []

    with open(fasta_path, "r") as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_string_protein_id is not None:
                    sequence = "".join(sequence_fragments).strip().upper()
                    protein_sequences_by_string_id[current_string_protein_id] = sequence

                current_string_protein_id = line[1:].strip()
                sequence_fragments = []
            else:
                sequence_fragments.append(line)

    if current_string_protein_id is not None:
        sequence = "".join(sequence_fragments).strip().upper()
        protein_sequences_by_string_id[current_string_protein_id] = sequence

    return protein_sequences_by_string_id


def extract_ensp_numeric_id(string_protein_id: str) -> int:
    """
    For deterministic sorting when multiple ENSP ids exist.
    Example: '9606.ENSP00000000233' -> 233
    Fallback: large number if parse fails.
    """
    try:
        if "ENSP" in string_protein_id:
            numeric_part = string_protein_id.split("ENSP", 1)[1]
            return int(numeric_part.lstrip("0") or "0")
        return int(string_protein_id)
    except Exception:
        return 10**18


def get_candidate_string_protein_ids_for_gene(
    gene_symbol: str,
    string_protein_ids_by_gene: Dict[str, Set[str]],
) -> Set[str]:
    """
    Tries exact match, then upper, then lower, to be robust.
    """
    if gene_symbol in string_protein_ids_by_gene:
        return set(string_protein_ids_by_gene[gene_symbol])

    gene_symbol_upper = gene_symbol.upper()
    if gene_symbol_upper in string_protein_ids_by_gene:
        return set(string_protein_ids_by_gene[gene_symbol_upper])

    gene_symbol_lower = gene_symbol.lower()
    if gene_symbol_lower in string_protein_ids_by_gene:
        return set(string_protein_ids_by_gene[gene_symbol_lower])

    return set()


def choose_best_string_protein_id(
    candidate_string_protein_id_set: Set[str],
    protein_sequences_by_string_id: Dict[str, str],
) -> Tuple[Optional[str], bool]:
    """
    Returns:
        (chosen_id, was_ambiguous)

    Selection rule:
      1) Prefer IDs that exist in FASTA
      2) Among them, choose smallest ENSP numeric id
      3) If none exist in FASTA, still choose smallest ENSP numeric id
    """
    if not candidate_string_protein_id_set:
        return None, False

    candidate_id_list = sorted(
        candidate_string_protein_id_set,
        key=lambda value: extract_ensp_numeric_id(value),
    )

    candidate_id_list_in_fasta = [value for value in candidate_id_list if value in protein_sequences_by_string_id]
    was_ambiguous = len(candidate_string_protein_id_set) > 1

    if candidate_id_list_in_fasta:
        return candidate_id_list_in_fasta[0], was_ambiguous
    return candidate_id_list[0], was_ambiguous


def compute_mean_pooled_esm3_embedding(
    esm3_client: ESM3,
    amino_acid_sequence: str,
    device: torch.device,
) -> np.ndarray:
    """
    Produces a 1D embedding vector by mean pooling per-residue embeddings.
    Returns:
        numpy float32 array of shape (embedding_dimension,)
    """
    protein_object = ESMProtein(sequence=amino_acid_sequence)
    protein_tensor = esm3_client.encode(protein_object)

    if hasattr(protein_tensor, "to"):
        protein_tensor = protein_tensor.to(device)

    with torch.no_grad():
        output = esm3_client.forward_and_sample(
            protein_tensor,
            SamplingConfig(return_per_residue_embeddings=True),
        )

        per_residue_embedding = output.per_residue_embedding
        if per_residue_embedding.dim() == 3:
            per_residue_embedding = per_residue_embedding[0]

        mean_pooled_embedding = per_residue_embedding.mean(dim=0)
        embedding_vector = mean_pooled_embedding.detach().to("cpu").float().numpy()

    del output
    del protein_tensor

    if device.type == "cuda":
        torch.cuda.empty_cache()
    else:
        gc.collect()

    return embedding_vector


def write_truncation_tsv(truncation_tsv_path: str, truncated_records: List[Tuple[str, str, int, int]]) -> None:
    """
    Columns:
      gene, string_protein_id, original_length, truncated_length
    """
    os.makedirs(os.path.dirname(os.path.abspath(truncation_tsv_path)), exist_ok=True)
    with open(truncation_tsv_path, "w", newline="") as file_handle:
        writer = csv.writer(file_handle, delimiter="\t")
        writer.writerow(["gene", "string_protein_id", "original_length", "truncated_length"])
        for gene_symbol, string_protein_id, original_length, truncated_length in truncated_records:
            writer.writerow([gene_symbol, string_protein_id, original_length, truncated_length])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppi", type=str, required=True, help="Path to ppi_network.tsv (protein1, protein2).")
    parser.add_argument("--mapping", type=str, required=True, help="Path to mapping.tsv (string_protein_id, gene).")
    parser.add_argument("--fasta", type=str, required=True, help="Path to protein FASTA (>string_protein_id).")
    parser.add_argument("--output", type=str, default="./gene_embedding/ESM3_embeddings.pkl", help="Output pickle path")
    parser.add_argument("--device", type=str, default="cuda", help="cuda, cuda:0, or cpu (default: cuda)")
    parser.add_argument("--max-len", type=int, default=2048, help="Max AA length (truncate if longer). default: 2048")
    parser.add_argument(
        "--truncation-tsv",
        type=str,
        default=None,
        help="Where to save truncation log TSV. Default: <output_without_ext>.truncated.tsv",
    )
    args = parser.parse_args()

    # Resolve device
    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        if args.device.startswith("cuda"):
            print("CUDA not available; falling back to CPU.")
        device = torch.device("cpu")

    output_path = os.path.abspath(args.output)
    output_prefix = os.path.splitext(output_path)[0]
    truncation_tsv_path = args.truncation_tsv or f"{output_prefix}.truncated.tsv"

    max_amino_acid_length: Optional[int] = args.max_len

    print("[1/5] Loading PPI nodes (unique genes) from ppi_network.tsv...")
    unique_gene_symbol_list = load_unique_gene_symbols_from_ppi_network(args.ppi)
    print(f"  unique nodes: {len(unique_gene_symbol_list)}")

    print("[2/5] Loading mapping.tsv (gene -> string_protein_id set)...")
    string_protein_ids_by_gene = load_string_protein_ids_by_gene(args.mapping)
    print(f"  genes in mapping: {len(string_protein_ids_by_gene)}")

    print("[3/5] Loading FASTA sequences (string_protein_id -> AA sequence)...")
    protein_sequences_by_string_id = load_fasta_sequences_by_string_id(args.fasta)
    print(f"  sequences loaded: {len(protein_sequences_by_string_id)}")

    print("[4/5] Loading ESM3 model...")
    esm3_client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=device)

    embeddings_by_gene_symbol: Dict[str, np.ndarray] = {}
    unresolved_gene_symbols: List[str] = []
    unresolved_sequences: List[str] = []
    ambiguous_mappings: List[Tuple[str, List[str], str]] = []
    truncated_records: List[Tuple[str, str, int, int]] = []

    print("[5/5] Computing embeddings...")
    for gene_symbol in tqdm(unique_gene_symbol_list, desc="ESM3 embeddings"):
        candidate_string_protein_id_set = get_candidate_string_protein_ids_for_gene(
            gene_symbol=gene_symbol,
            string_protein_ids_by_gene=string_protein_ids_by_gene,
        )

        if not candidate_string_protein_id_set:
            unresolved_gene_symbols.append(gene_symbol)
            continue

        chosen_string_protein_id, was_ambiguous = choose_best_string_protein_id(
            candidate_string_protein_id_set=candidate_string_protein_id_set,
            protein_sequences_by_string_id=protein_sequences_by_string_id,
        )

        if chosen_string_protein_id is None:
            unresolved_gene_symbols.append(gene_symbol)
            continue

        if was_ambiguous:
            ambiguous_mappings.append(
                (gene_symbol, sorted(list(candidate_string_protein_id_set)), chosen_string_protein_id)
            )

        amino_acid_sequence = protein_sequences_by_string_id.get(chosen_string_protein_id)
        if not amino_acid_sequence:
            unresolved_sequences.append(f"{gene_symbol}\t{chosen_string_protein_id}\terror=missing_in_fasta")
            continue

        amino_acid_sequence = amino_acid_sequence.replace(" ", "").replace("\t", "").replace("\r", "").upper()
        original_length = len(amino_acid_sequence)

        if max_amino_acid_length is not None and original_length > max_amino_acid_length:
            amino_acid_sequence = amino_acid_sequence[:max_amino_acid_length]
            truncated_records.append((gene_symbol, chosen_string_protein_id, original_length, len(amino_acid_sequence)))

        try:
            embedding_vector = compute_mean_pooled_esm3_embedding(
                esm3_client=esm3_client,
                amino_acid_sequence=amino_acid_sequence,
                device=device,
            )
        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            unresolved_sequences.append(
                f"{gene_symbol}\t{chosen_string_protein_id}\tlen={original_length}\terror=oom_even_after_truncation"
            )
            continue

        embeddings_by_gene_symbol[gene_symbol] = embedding_vector

    # Save dictionary: {gene_symbol: embedding_vector}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as file_handle:
        pickle.dump(embeddings_by_gene_symbol, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save truncation TSV
    write_truncation_tsv(truncation_tsv_path, truncated_records)

    print("\nDone.")
    print(f"Saved embeddings pickle: {output_path}")
    print(f"Saved truncation TSV:    {truncation_tsv_path}")
    print(f"Embeddings computed: {len(embeddings_by_gene_symbol)} / {len(unique_gene_symbol_list)}")
    print(f"Truncated sequences: {len(truncated_records)} (max_len={max_amino_acid_length})")
    print(f"Unresolved genes (no mapping entry): {len(unresolved_gene_symbols)}")
    print(f"Unresolved sequences (missing in FASTA / OOM / other errors): {len(unresolved_sequences)}")
    print(f"Ambiguous mappings (multiple candidate string IDs): {len(ambiguous_mappings)}")

    # Debug logs next to output
    if unresolved_gene_symbols:
        with open(f"{output_prefix}.unresolved_genes.txt", "w") as file_handle:
            file_handle.write("\n".join(unresolved_gene_symbols) + "\n")

    if unresolved_sequences:
        with open(f"{output_prefix}.unresolved_sequences.txt", "w") as file_handle:
            file_handle.write("\n".join(unresolved_sequences) + "\n")

    if ambiguous_mappings:
        with open(f"{output_prefix}.ambiguous_mappings.tsv", "w") as file_handle:
            file_handle.write("gene\tcandidate_string_protein_ids\tchosen_string_protein_id\n")
            for gene_symbol, candidate_id_list, chosen_id in ambiguous_mappings:
                file_handle.write(f"{gene_symbol}\t{','.join(candidate_id_list)}\t{chosen_id}\n")


if __name__ == "__main__":
    main()
