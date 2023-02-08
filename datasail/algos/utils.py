from typing import List


def estimate_surviving_interactions(num_inter: int, num_drugs: int, num_proteins: int, splits: List[float]) -> int:
    sparsity = num_inter / (num_drugs * num_proteins)
    dense_survivors = sum(s ** 2 for s in splits) * num_drugs * num_proteins
    return int(dense_survivors * sparsity + 0.5)
