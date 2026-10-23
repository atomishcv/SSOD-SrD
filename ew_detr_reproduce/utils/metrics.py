"""
EWOD Evaluation Metrics
========================
Paper: EW-DETR, Section 4.2 and Appendix D.

Implements the three sub-scores and the composite FOGS score:

  FSS  — Forgetting Sub-Score      (Eq. 12)
  OSS  — Openness Sub-Score        (Eq. 13)
  GSS  — Generalisation Sub-Score  (Eq. 14)
  FOGS = mean(FSS, OSS, GSS)
"""

from __future__ import annotations
from typing import Dict, List


def fss(
    map_prev_per_task: List[float],
    map_curr_when_introduced: List[float],
) -> float:
    """
    Forgetting Sub-Score (FSS).

    FSS = (1 / (T-1)) * Σ_{t=1}^{T-1} [ mAP_prev_t / Avg(mAP_curr_{1:t}) ]

    Higher FSS → less forgetting.

    Parameters
    ----------
    map_prev_per_task : List[float]
        For each task t (1 to T-1), the mAP on previously learned classes at
        the *final* task T.
    map_curr_when_introduced : List[float]
        For each task t (1 to T-1), the average mAP on classes introduced in
        tasks 1..t when each of those tasks was first encountered.

    Returns
    -------
    float : FSS score (higher is better, max ~100 for perfect retention).
    """
    if len(map_prev_per_task) == 0:
        return 0.0
    assert len(map_prev_per_task) == len(map_curr_when_introduced)

    scores = []
    for map_prev, map_init_avg in zip(map_prev_per_task, map_curr_when_introduced):
        if map_init_avg == 0.0:
            scores.append(0.0)
        else:
            scores.append(map_prev / map_init_avg * 100.0)

    return sum(scores) / len(scores)


def oss(
    u_recall_per_task: List[float],
    wi_per_task: List[float],
    a_ose_per_task: List[float],
    gt_unk_per_task: List[float],
) -> float:
    """
    Openness Sub-Score (OSS).

    OSS = (1/T) Σ_t [ U-Recall_t + (1 - WI_t) + 1 / (1 + A-OSE_t / GT_unk_t) ]
                / 3

    All per-task values should be in [0, 1] (proportions, NOT percentages).

    Parameters
    ----------
    u_recall_per_task   : Unknown recall per task.
    wi_per_task         : Wilderness Impact per task (lower is better → we use 1-WI).
    a_ose_per_task      : Absolute Open-Set Error per task.
    gt_unk_per_task     : Ground-truth unknown count per task.

    Returns
    -------
    float : OSS score (higher is better, max ~100).
    """
    T = len(u_recall_per_task)
    if T == 0:
        return 0.0
    if not (
        len(wi_per_task) == T
        and len(a_ose_per_task) == T
        and len(gt_unk_per_task) == T
    ):
        raise ValueError("OSS inputs must have the same number of tasks.")

    total = 0.0
    for t in range(T):
        u_rec = u_recall_per_task[t]
        wi    = wi_per_task[t]
        a_ose = a_ose_per_task[t]
        gt    = gt_unk_per_task[t]

        normalised_a_ose = 1.0 / (1.0 + a_ose / max(gt, 1e-6))
        total += (u_rec + (1.0 - wi) + normalised_a_ose) / 3.0

    return (total / T) * 100.0


def gss(map_curr_per_task: List[float]) -> float:
    """
    Generalisation Sub-Score (GSS).

    GSS = (1 / (T-1)) Σ_{t=2}^{T} mAP_curr_t

    Measures how well current-task classes transfer across all seen domains.

    Parameters
    ----------
    map_curr_per_task : List[float]
        mAP on current-task classes when evaluated across all seen domains,
        for tasks t = 2, ..., T (i.e., length T-1).

    Returns
    -------
    float : GSS score (higher is better).
    """
    if len(map_curr_per_task) == 0:
        return 0.0
    return sum(map_curr_per_task) / len(map_curr_per_task)


def fogs(fss_score: float, oss_score: float, gss_score: float) -> float:
    """
    FOGS = (FSS + OSS + GSS) / 3

    Holistic composite metric balancing forgetting resistance, openness, and
    generalisation.
    """
    return (fss_score + oss_score + gss_score) / 3.0


def compute_all_scores(
    map_prev_per_task: List[float],
    map_curr_when_introduced: List[float],
    u_recall_per_task: List[float],
    wi_per_task: List[float],
    a_ose_per_task: List[float],
    gt_unk_per_task: List[float],
    map_curr_generalisation: List[float],
) -> Dict[str, float]:
    """
    Convenience function to compute all four EWOD metrics at once.

    Returns
    -------
    dict with keys: 'FSS', 'OSS', 'GSS', 'FOGS'
    """
    fss_val  = fss(map_prev_per_task, map_curr_when_introduced)
    oss_val  = oss(u_recall_per_task, wi_per_task, a_ose_per_task, gt_unk_per_task)
    gss_val  = gss(map_curr_generalisation)
    fogs_val = fogs(fss_val, oss_val, gss_val)

    return {
        "FSS":  round(fss_val,  4),
        "OSS":  round(oss_val,  4),
        "GSS":  round(gss_val,  4),
        "FOGS": round(fogs_val, 4),
    }
