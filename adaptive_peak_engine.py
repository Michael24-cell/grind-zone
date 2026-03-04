"""
adaptive_peak_engine.py
-----------------------
Pharmacokinetic adaptive peak calibration engine.

Takes a list of feel-score log entries and returns a personalised peak
shift, confidence score, and adjusted peak time for a given drug.

No UI. No network calls. Pure logic.

Usage example
-------------
from adaptive_peak_engine import compute_adaptive_peak, DRUG_DEFAULTS

entries = [
    {"elapsed_hours": 1.4, "feel_score": 7, "timestamp": "2024-01-01T09:00:00"},
    {"elapsed_hours": 1.6, "feel_score": 9, "timestamp": "2024-01-02T09:10:00"},
    {"elapsed_hours": 1.8, "feel_score": 8, "timestamp": "2024-01-03T09:05:00"},
]

result = compute_adaptive_peak(entries, drug="ritalin")
print(result)
"""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUCKET_INTERVAL = 0.2  # hours — every 12 minutes

PHANTOM_COUNT = 8       # number of synthetic anchor entries injected at default peak
PHANTOM_FEEL = 8        # feel score assigned to each phantom anchor
DECAY_RATE = 0.88       # exponential decay weight per older entry (newest = 1.0)
PEAK_THRESHOLD = 0.85   # include buckets within 85% of the best bucket score

MIN_ENTRIES_FOR_CALIBRATION = 6  # engine stays dormant below this real-entry count

# Per-drug defaults: (default_peak_hours, max_shift_hours)
# Source: FDA labels + clinical pharmacology literature cited in README
DRUG_DEFAULTS = {
    "ritalin":     {"default_peak": 1.5, "max_shift": 2.0},
    "concerta":    {"default_peak": 6.8, "max_shift": 2.0},
    "wellbutrin":  {"default_peak": 5.0, "max_shift": 2.0},
    "lamictal":    {"default_peak": 0.0, "max_shift": 1.5},  # cumulative — no acute peak
    "remeron":     {"default_peak": 1.0, "max_shift": 1.5},
    "lexapro":     {"default_peak": 5.0, "max_shift": 1.0},
}


# ---------------------------------------------------------------------------
# Step 1 — Bucketing
# ---------------------------------------------------------------------------

def bucket_elapsed(elapsed_hours: float, interval: float = BUCKET_INTERVAL) -> float:
    """
    Round an elapsed-hours value to the nearest bucket boundary.

    Example with interval=0.2:
        1.13  ->  1.2
        1.07  ->  1.0
        2.55  ->  2.6

    This groups log entries that are close in time so small differences
    in when someone logs don't create meaningless separate buckets.
    """
    return round(round(elapsed_hours / interval) * interval, 10)


# ---------------------------------------------------------------------------
# Step 2 — Phantom anchors
# ---------------------------------------------------------------------------

def build_phantom_anchors(default_peak: float) -> list:
    """
    Create PHANTOM_COUNT synthetic entries all sitting at the drug's
    published default peak with a feel score of PHANTOM_FEEL (8).

    These entries are prepended to the real log before any weighting.
    They act as a centre of gravity: real data must accumulate and agree
    before it can meaningfully pull the detected peak away from the
    pharmacokinetic default.

    Phantoms have no timestamp — they are always treated as the oldest
    entries (lowest decay weight).
    """
    return [
        {
            "elapsed_hours": default_peak,
            "feel_score": PHANTOM_FEEL,
            "timestamp": None,  # phantom — no real timestamp
            "is_phantom": True,
        }
        for _ in range(PHANTOM_COUNT)
    ]


# ---------------------------------------------------------------------------
# Step 3 — Exponential decay weighting
# ---------------------------------------------------------------------------

def apply_decay_weights(entries: list) -> list:
    """
    Assign a decay weight to each entry based on recency.

    Entries must be sorted oldest-first before calling this function.
    The newest entry (last in the list) gets weight 1.0.
    Each step further into the past multiplies by DECAY_RATE (0.88).

    Returns a new list of dicts with a 'weight' key added.

    Why exponential decay?
    ----------------------
    Your pharmacokinetics shift slowly over time — stress, diet, sleep,
    tolerance all nudge your personal curve. Decay ensures the engine
    tracks your *current* response rather than an average of all history.
    """
    n = len(entries)
    weighted = []
    for i, entry in enumerate(entries):
        # Distance from the newest entry (index n-1)
        age = (n - 1) - i
        weight = DECAY_RATE ** age
        weighted.append({**entry, "weight": weight})
    return weighted


# ---------------------------------------------------------------------------
# Step 4 — Bucket aggregation
# ---------------------------------------------------------------------------

def aggregate_buckets(weighted_entries: list) -> dict:
    """
    Group weighted entries by their bucketed elapsed-hours value.

    For each bucket, compute a weighted average feel score:
        bucket_score = sum(feel * weight) / sum(weight)

    Returns a dict mapping bucket_time -> weighted_average_feel_score.
    """
    bucket_feel_sum = {}    # bucket -> sum of (feel * weight)
    bucket_weight_sum = {}  # bucket -> sum of weights

    for entry in weighted_entries:
        b = bucket_elapsed(entry["elapsed_hours"])
        feel = entry["feel_score"]
        w = entry["weight"]

        bucket_feel_sum[b] = bucket_feel_sum.get(b, 0.0) + feel * w
        bucket_weight_sum[b] = bucket_weight_sum.get(b, 0.0) + w

    return {
        b: bucket_feel_sum[b] / bucket_weight_sum[b]
        for b in bucket_feel_sum
    }


# ---------------------------------------------------------------------------
# Step 5 — Peak centroid detection
# ---------------------------------------------------------------------------

def detect_peak_centroid(bucket_scores: dict) -> float:
    """
    Find the centre of mass of the peak region.

    Algorithm:
    1. Find the single highest-scoring bucket.
    2. Collect all buckets whose score is within PEAK_THRESHOLD (85%)
       of that best score. This handles broad, plateau-shaped peaks.
    3. Compute a score-weighted centroid across those top buckets.

    Centroid formula:
        centroid = sum(bucket_time * score) / sum(score)

    Returns the centroid elapsed-hours value.
    """
    if not bucket_scores:
        return 0.0

    best_score = max(bucket_scores.values())
    cutoff = best_score * PEAK_THRESHOLD

    top_buckets = {b: s for b, s in bucket_scores.items() if s >= cutoff}

    total_score = sum(top_buckets.values())
    centroid = sum(b * s for b, s in top_buckets.items()) / total_score

    return centroid


# ---------------------------------------------------------------------------
# Step 6 — Shift clamping
# ---------------------------------------------------------------------------

def clamp_shift(shift: float, max_shift: float) -> float:
    """
    Clamp the raw shift to the per-drug maximum.

    No matter what the data says, the engine will never push the peak
    outside pharmacokinetically plausible bounds. For example, Ritalin's
    first peak cannot be 6 hours in — that would contradict the drug's
    known absorption profile.
    """
    return max(-max_shift, min(max_shift, shift))


# ---------------------------------------------------------------------------
# Step 7 — Confidence score
# ---------------------------------------------------------------------------

def compute_confidence(real_entries: list) -> float:
    """
    Return a 0.0 -> 1.0 confidence score.

    Two signals contribute equally:
    1. Log count signal:    how many real entries exist vs a 'saturated'
                            count of 30 (beyond 30 logs, count stops mattering)
    2. High-feel signal:    proportion of real entries with feel >= 7
                            (high-feel entries carry the most signal about
                            where your actual peak is)

    Below MIN_ENTRIES_FOR_CALIBRATION the engine returns 0.0 —
    there is not enough data to trust any shift.
    """
    n = len(real_entries)
    if n < MIN_ENTRIES_FOR_CALIBRATION:
        return 0.0

    count_signal = min(n / 30.0, 1.0)

    high_feel = sum(1 for e in real_entries if e["feel_score"] >= 7)
    high_feel_signal = min(high_feel / 10.0, 1.0)

    return round((count_signal + high_feel_signal) / 2.0, 3)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_adaptive_peak(entries: list, drug: str) -> dict:
    """
    Run the full adaptive peak calibration pipeline.

    Parameters
    ----------
    entries : list of dicts, each containing:
        - elapsed_hours  (float)  hours since dose at time of log
        - feel_score     (int)    subjective score 1-10
        - timestamp      (str)    ISO 8601 string, used for recency sorting

    drug : str
        One of the keys in DRUG_DEFAULTS (e.g. "ritalin", "wellbutrin").

    Returns
    -------
    dict with keys:
        - shift          (float)  hours the peak moved from default (+ = later)
        - confidence     (float)  0.0 to 1.0
        - adjusted_peak  (float)  default_peak + shift
        - default_peak   (float)  the drug's published pharmacokinetic default
        - entry_count    (int)    number of real entries provided
        - calibrated     (bool)   False if below minimum entry threshold
    """
    drug = drug.lower()
    if drug not in DRUG_DEFAULTS:
        raise ValueError(
            f"Unknown drug '{drug}'. Valid options: {list(DRUG_DEFAULTS.keys())}"
        )

    default_peak = DRUG_DEFAULTS[drug]["default_peak"]
    max_shift = DRUG_DEFAULTS[drug]["max_shift"]

    # --- Confidence check first ---
    confidence = compute_confidence(entries)
    if confidence == 0.0:
        return {
            "shift": 0.0,
            "confidence": 0.0,
            "adjusted_peak": default_peak,
            "default_peak": default_peak,
            "entry_count": len(entries),
            "calibrated": False,
        }

    # --- Sort real entries oldest-first by timestamp ---
    real_sorted = sorted(
        entries,
        key=lambda e: e["timestamp"] if e["timestamp"] else "",
    )

    # --- Prepend phantom anchors (they are 'older' than all real entries) ---
    phantoms = build_phantom_anchors(default_peak)
    all_entries = phantoms + real_sorted

    # --- Apply exponential decay weights ---
    weighted = apply_decay_weights(all_entries)

    # --- Aggregate into 0.2-hour buckets ---
    bucket_scores = aggregate_buckets(weighted)

    # --- Find peak centroid ---
    centroid = detect_peak_centroid(bucket_scores)

    # --- Compute shift and clamp ---
    raw_shift = centroid - default_peak
    shift = clamp_shift(raw_shift, max_shift)

    adjusted_peak = round(default_peak + shift, 3)

    return {
        "shift": round(shift, 3),
        "confidence": confidence,
        "adjusted_peak": adjusted_peak,
        "default_peak": default_peak,
        "entry_count": len(entries),
        "calibrated": True,
    }


# ---------------------------------------------------------------------------
# Simulations (run: python adaptive_peak_engine.py)
# ---------------------------------------------------------------------------

import random


# The "real" personal peak we want the engine to discover
TRUE_PEAK = 1.8
SIM_DRUG = "ritalin"


def make_entry(elapsed, feel, day):
    """Build one log entry dict."""
    return {
        "elapsed_hours": round(max(0.1, elapsed), 3),
        "feel_score": max(1, min(10, round(feel))),
        "timestamp": f"2024-01-{day:02d}T09:00:00",
    }


def generate_clean_logs(n=30):
    """
    Synthetic logs with a clear peak at TRUE_PEAK.
    Feel score is highest when elapsed is closest to TRUE_PEAK.
    """
    logs = []
    for i in range(n):
        elapsed = 0.4 + (i % 15) * 0.2
        feel = 9 - abs(elapsed - TRUE_PEAK) * 3
        logs.append(make_entry(elapsed, feel, i + 1))
    return logs


def generate_noisy_logs(n=30, seed=None):
    """Same shape as clean logs but with random noise on both axes."""
    if seed is not None:
        random.seed(seed)
    logs = []
    for i in range(n):
        elapsed = 0.4 + (i % 15) * 0.2
        feel = 9 - abs(elapsed - TRUE_PEAK) * 3
        noisy_feel = feel + random.uniform(-1.0, 1.0)
        noisy_time = elapsed + random.uniform(-0.3, 0.3)
        logs.append(make_entry(noisy_time, noisy_feel, i + 1))
    return logs


# --- Experiment 1: Convergence ---

def experiment_convergence():
    print("=" * 60)
    print("EXPERIMENT 1 — Convergence")
    print(f"True peak: {TRUE_PEAK}h  |  Default peak: {DRUG_DEFAULTS[SIM_DRUG]['default_peak']}h")
    print("=" * 60)

    logs = generate_clean_logs(30)

    for i in range(3, len(logs) + 1):
        subset = logs[:i]
        result = compute_adaptive_peak(subset, drug=SIM_DRUG)
        peak = result["adjusted_peak"]
        conf = result["confidence"]
        status = "(not calibrated yet)" if not result["calibrated"] else ""
        print(f"  Logs: {i:>2}  | Peak: {peak:.2f}h  | Conf: {conf:.3f}  {status}")

    print()


# --- Experiment 2: Noise testing ---

def experiment_noise():
    print("=" * 60)
    print("EXPERIMENT 2 — Noise testing  (50 runs)")
    print(f"True peak: {TRUE_PEAK}h  |  Noise: ±1 feel point, ±0.3h elapsed")
    print("=" * 60)

    estimates = []
    for run in range(50):
        logs = generate_noisy_logs(n=30, seed=run)
        result = compute_adaptive_peak(logs, drug=SIM_DRUG)
        estimates.append(result["adjusted_peak"])

    avg = sum(estimates) / len(estimates)
    close = sum(1 for e in estimates if abs(e - TRUE_PEAK) <= 0.2)

    print(f"  Mean estimate:   {avg:.3f}h")
    print(f"  Min / Max:       {min(estimates):.3f}h / {max(estimates):.3f}h")
    print(f"  Within ±0.2h:    {close}/50 runs")
    print()


# --- Experiment 3: Parameter sensitivity ---

def experiment_parameters():
    print("=" * 60)
    print("EXPERIMENT 3 — Parameter sensitivity")
    print(f"True peak: {TRUE_PEAK}h  |  30 clean logs")
    print("=" * 60)

    logs = generate_clean_logs(30)

    # We temporarily patch the module-level constants to test sensitivity.
    # This is the simplest way to do it without restructuring the functions.
    import adaptive_peak_engine as _eng

    original_threshold = _eng.PEAK_THRESHOLD
    original_decay = _eng.DECAY_RATE

    print("\n  -- Peak threshold (decay fixed at 0.88) --")
    for threshold in [0.75, 0.80, 0.85, 0.90, 0.95]:
        _eng.PEAK_THRESHOLD = threshold
        result = compute_adaptive_peak(logs, drug=SIM_DRUG)
        print(f"  Threshold: {threshold:.2f}  | Peak: {result['adjusted_peak']:.3f}h")

    _eng.PEAK_THRESHOLD = original_threshold  # restore

    print("\n  -- Decay rate (threshold fixed at 0.85) --")
    for decay in [0.70, 0.80, 0.88, 0.95, 1.00]:
        _eng.DECAY_RATE = decay
        result = compute_adaptive_peak(logs, drug=SIM_DRUG)
        label = "(no decay — all entries equal weight)" if decay == 1.00 else ""
        print(f"  Decay:     {decay:.2f}  | Peak: {result['adjusted_peak']:.3f}h  {label}")

    _eng.DECAY_RATE = original_decay  # restore

    print()


# --- Experiment 4: ASCII chart ---

def experiment_ascii_chart():
    print("=" * 60)
    print("EXPERIMENT 4 — ASCII convergence chart")
    print(f"True peak: {TRUE_PEAK}h  |  Each █ = 0.05h")
    print("=" * 60)
    print(f"\n  {'Logs':<6} {'Peak':>6}   Chart\n")

    logs = generate_clean_logs(30)

    for i in range(3, len(logs) + 1):
        subset = logs[:i]
        result = compute_adaptive_peak(subset, drug=SIM_DRUG)
        peak = result["adjusted_peak"]
        bar = "█" * int(peak / 0.05)
        print(f"  {i:<6} {peak:>5.2f}h  {bar}")

    print()


# --- Run all experiments ---

if __name__ == "__main__":
    experiment_convergence()
    experiment_noise()
    experiment_parameters()
    experiment_ascii_chart()
