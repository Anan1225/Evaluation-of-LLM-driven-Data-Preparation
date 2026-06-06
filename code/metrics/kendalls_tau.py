from itertools import combinations
from typing import List, Union

def digits_to_perm(x: Union[int, str]) -> List[int]:
    """
    Convert 12345 or "12345" -> [1,2,3,4,5]
    If you have multi-digit items (e.g., 10,11), do NOT use this format.
    """
    s = str(x).strip()
    if not s.isdigit():
        raise ValueError(f"Input must be digits only, got: {x!r}")
    return [int(ch) for ch in s]

def kendalls_tau_from_perms(
    gt: Union[int, str, List[int]],
    pred: Union[int, str, List[int]],
) -> float:
    """
    Kendall's tau (no ties) for two permutations of the same items.
    tau = (C - D) / (n*(n-1)/2)
    where C = #concordant pairs, D = #discordant pairs.
    """
    # Parse inputs
    if isinstance(gt, (int, str)):
        gt_perm = digits_to_perm(gt)
    else:
        gt_perm = list(gt)

    if isinstance(pred, (int, str)):
        pred_perm = digits_to_perm(pred)
    else:
        pred_perm = list(pred)

    if len(gt_perm) != len(pred_perm):
        raise ValueError("gt and pred must have the same length")

    n = len(gt_perm)
    if n < 2:
        return 0.0

    # Validate they are permutations of the same items
    if sorted(gt_perm) != sorted(pred_perm):
        raise ValueError("gt and pred must contain the same items (a permutation)")

    # Map item -> rank in each permutation
    rank_gt = {item: i for i, item in enumerate(gt_perm)}
    rank_pred = {item: i for i, item in enumerate(pred_perm)}

    concordant = 0
    discordant = 0

    items = gt_perm  # or pred_perm, same set
    for a, b in combinations(items, 2):
        # Compare relative order in gt vs pred
        s_gt = rank_gt[a] - rank_gt[b]
        s_pred = rank_pred[a] - rank_pred[b]
        if s_gt * s_pred > 0:
            concordant += 1
        else:
            discordant += 1

    total_pairs = n * (n - 1) // 2
    return (concordant - discordant) / total_pairs

# Example
# song name 1
# artist name 2
# album name 3
# time 4
# release year 5
gt = 12354
pred = 21354
print(kendalls_tau_from_perms(gt, pred))
