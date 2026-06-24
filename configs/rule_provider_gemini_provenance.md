# Gemini Rule Provider Provenance

This rule provider was generated with the free Gemini web interface from the prompt below.

## Prompt

```text
I am building a weakly aligned text-tabular benchmark for synthetic data generation.

Tabular dataset:
- Source: Bank Marketing
- Target column: target
- target=1 means the customer subscribed / positive marketing response
- target=0 means no subscription / negative response
- Target distribution:
  - target=0: 39922 rows, 88.3%
  - target=1: 5289 rows, 11.7%

Text dataset:
- Source: FinancialPhraseBank
- Text label column: sentiment
- Available labels:
  - neutral: 2873 rows, 59.36%
  - positive: 1363 rows, 28.16%
  - negative: 604 rows, 12.48%

Task:
Generate a top-down rule-provider JSON for weakly aligning the tabular target with text sentiment.

Requirements:
- Return only valid JSON.
- Use this exact schema:
{
  "name": "...",
  "description": "...",
  "schema": {
    "tabular_source": "bank_marketing",
    "text_source": "financial_phrasebank",
    "target_column": "target",
    "text_label_column": "sentiment"
  },
  "alignment_rules": {
    "target_to_text_sentiment": {
      "0": [...],
      "1": [...]
    }
  },
  "constraints": {
    "preserve_target_distribution": true,
    "sample_text_with_replacement": true
  },
  "rationale": "..."
}

Use only these sentiment labels: positive, neutral, negative.
The rule should be semantically plausible: target=1 should align with more favorable financial sentiment, while target=0 should align with neutral or unfavorable sentiment.
```

## Gemini Output

The generated JSON is saved in:

- `configs/rule_provider_gemini.json`
