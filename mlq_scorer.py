"""
MLQ (Multifactor Leadership Questionnaire) subscale scorer.
Bass & Avolio (1995) scoring: 0-4 Likert per item.
"""

# Item assignments per Bass & Avolio MLQ Form 5X-Short
SUBSCALES = {
    "IIA": {"name": "Idealized Influence (Attributed)", "items": [10, 18, 21, 25], "type": "Transformational"},
    "IIB": {"name": "Idealized Influence (Behaviour)", "items": [6, 14, 23, 34], "type": "Transformational"},
    "IM":  {"name": "Inspirational Motivation", "items": [9, 13, 26, 36], "type": "Transformational"},
    "IS":  {"name": "Intellectual Stimulation", "items": [2, 8, 30, 32], "type": "Transformational"},
    "IC":  {"name": "Individualised Consideration", "items": [15, 19, 29, 31], "type": "Transformational"},
    "CR":  {"name": "Contingent Reward", "items": [1, 11, 16, 35], "type": "Transactional"},
    "MBEA":{"name": "Management by Exception (Active)", "items": [4, 22, 24, 27], "type": "Transactional"},
    "MBEP":{"name": "Management by Exception (Passive)", "items": [3, 12, 17, 20], "type": "Passive/Avoidant"},
    "LF":  {"name": "Laissez-Faire", "items": [5, 7, 28, 33], "type": "Passive/Avoidant"},
}

COMPOSITES = {
    "Transformational": ["IIA", "IIB", "IM", "IS", "IC"],
    "Transactional": ["CR", "MBEA"],
    "PassiveAvoidant": ["MBEP", "LF"],
}


def score_responses(df):
    """
    Score an MLQ dataframe.
    Expects columns named Q1 through Q45 (0-4 integer values).
    Returns df with subscale and composite columns appended.
    """
    import pandas as pd

    result = df.copy()

    # Subscale scores
    for code, meta in SUBSCALES.items():
        cols = [f"Q{i}" for i in meta["items"]]
        missing = [c for c in cols if c not in result.columns]
        if missing:
            raise ValueError(f"Missing columns for {code}: {missing}")
        result[f"{code}_score"] = result[cols].sum(axis=1)

    # Composite scores
    for comp, codes in COMPOSITES.items():
        subscale_cols = [f"{c}_score" for c in codes]
        result[f"{comp}_total"] = result[subscale_cols].sum(axis=1)

    return result


def validate_responses(df):
    """Check for out-of-range values and missing data."""
    issues = []
    for col in [f"Q{i}" for i in range(1, 46)]:
        if col not in df.columns:
            continue
        oob = df[(df[col] < 0) | (df[col] > 4)]
        if not oob.empty:
            issues.append(f"{col}: {len(oob)} out-of-range values")
        nas = df[col].isna().sum()
        if nas > 0:
            issues.append(f"{col}: {nas} missing values")
    return issues
