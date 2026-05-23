"""
Regression analysis for MLQ subscale scores.
Reads scored CSV, runs linear/logistic regression, outputs results table.
"""

import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
from mlq_scorer import score_responses, validate_responses


def run_regression(csv_path: str, outcome_col: str, predictors: list = None, model_type: str = "linear"):
    """
    Run regression of outcome on MLQ composite scores.

    Args:
        csv_path: Path to raw MLQ export CSV
        outcome_col: Column name of the outcome variable
        predictors: List of predictor column names (default: all three composites)
        model_type: "linear" or "logistic"

    Returns:
        statsmodels results object
    """
    df = pd.read_csv(csv_path)

    # Validate
    issues = validate_responses(df)
    if issues:
        print("Validation warnings:")
        for issue in issues:
            print(f"  - {issue}")

    # Score
    df = score_responses(df)

    if predictors is None:
        predictors = ["Transformational_total", "Transactional_total", "PassiveAvoidant_total"]

    missing_cols = [c for c in predictors + [outcome_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    X = df[predictors].dropna()
    y = df.loc[X.index, outcome_col]
    X = sm.add_constant(X)

    if model_type == "linear":
        model = sm.OLS(y, X).fit()
    elif model_type == "logistic":
        model = sm.Logit(y, X).fit()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model


def format_results_table(result) -> pd.DataFrame:
    """Format statsmodels result into a clean summary table."""
    summary = pd.DataFrame({
        "B": result.params,
        "SE": result.bse,
        "t": result.tvalues,
        "p-value": result.pvalues,
        "95% CI Lower": result.conf_int()[0],
        "95% CI Upper": result.conf_int()[1],
    }).round(3)
    summary["Sig"] = summary["p-value"].apply(
        lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    )
    return summary


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "sample_mlq_data.csv"
    outcome = sys.argv[2] if len(sys.argv) > 2 else "outcome_score"
    result = run_regression(csv_path, outcome)
    print(result.summary())
    tbl = format_results_table(result)
    print("\nClean results table:")
    print(tbl.to_string())
    tbl.to_csv("regression_results.csv", index=True)
    print("\nSaved to regression_results.csv")
