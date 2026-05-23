"""
Entry point: score MLQ export and run regression analysis.
Usage: python main.py <input_csv> <outcome_column>
"""

import sys
import pandas as pd
from mlq_scorer import score_responses, validate_responses
from regression_analysis import run_regression, format_results_table


def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <input_csv> <outcome_column>")
        print("Example: python main.py mlq_export.csv employee_satisfaction")
        sys.exit(1)

    csv_path = sys.argv[1]
    outcome_col = sys.argv[2]
    model_type = sys.argv[3] if len(sys.argv) > 3 else "linear"

    print(f"MLQ Regression Pipeline")
    print(f"Input: {csv_path} | Outcome: {outcome_col} | Model: {model_type}")
    print("-" * 60)

    result = run_regression(csv_path, outcome_col, model_type=model_type)
    tbl = format_results_table(result)

    print(f"R² = {result.rsquared:.3f}" if hasattr(result, "rsquared") else "")
    print(tbl.to_string())
    tbl.to_csv("regression_results.csv", index=True)
    print("\nResults saved to regression_results.csv")


if __name__ == "__main__":
    main()
