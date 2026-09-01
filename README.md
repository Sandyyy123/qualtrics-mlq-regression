> **⚠️ Proprietary — All Rights Reserved.** © 2026 Sandeep Grover. This repository is licensed to Sandeep Grover and may **not** be used, run, copied, modified, distributed, or used to train models without prior written permission. Public visibility does not grant a license. See [LICENSE](LICENSE).

---

# Qualtrics MLQ Regression Pipeline

Regression-ready scoring and analysis for the **Multifactor Leadership Questionnaire (MLQ)** — Bass & Avolio Form 5X-Short.

## Architecture

```
Qualtrics Export (CSV)
        |
        v
[mlq_scorer.py] -- Subscale scoring (9 subscales + 3 composites)
        |
        v
[regression_analysis.py] -- Linear / Logistic regression via statsmodels
        |
        v
regression_results.csv + console summary
```

## Subscales Scored

| Code | Subscale | Type |
|------|----------|------|
| IIA | Idealized Influence (Attributed) | Transformational |
| IIB | Idealized Influence (Behaviour) | Transformational |
| IM | Inspirational Motivation | Transformational |
| IS | Intellectual Stimulation | Transformational |
| IC | Individualised Consideration | Transformational |
| CR | Contingent Reward | Transactional |
| MBEA | Management by Exception (Active) | Transactional |
| MBEP | Management by Exception (Passive) | Passive/Avoidant |
| LF | Laissez-Faire | Passive/Avoidant |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py mlq_export.csv outcome_score
python main.py mlq_export.csv outcome_score logistic
```

Input CSV must have columns `Q1` through `Q45` (0-4 integer values per MLQ Likert scale).

## Output

- Console: regression summary table (B, SE, t, p, 95% CI, significance stars)
- `regression_results.csv`: clean results table ready for reporting

## Qualtrics Setup Notes

The companion Qualtrics survey is configured with:
- Embedded data fields computing all 9 subscale scores in real-time
- Validation to reject out-of-range responses
- Export column headers matching this pipeline (`Q1`-`Q45` + subscale fields)

---
Built for Upwork client proposal - Dr. Sandeep Grover, PhD Data Scientist
