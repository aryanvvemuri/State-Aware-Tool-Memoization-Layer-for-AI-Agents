# Axiom Semantic Threshold Calibration Report

- **Embedding Model:** `all-MiniLM-L6-v2`
- **Calibrated Default Threshold:** `0.75`
- **Equivalent Pairs Mean Similarity:** `0.8584` (Min: `0.6611`, Max: `0.9480`)
- **Near-Miss Pairs Mean Similarity:** `0.7927` (Min: `0.5107`, Max: `0.9847`)
- **Unrelated Pairs Mean Similarity:** `0.0265` (Min: `-0.1836`, Max: `0.1881`)

## Threshold Sweep & Performance Metrics

| Threshold | TPR (Equivalent Recall) | FPR (Unrelated Noise) | Near-Miss Semantic Passthrough |
| :--- | :--- | :--- | :--- |
| `0.65` | `100.0%` | `0.0%` | `80.0%` |
| `0.70` | `95.0%` | `0.0%` | `75.0%` |
| `0.75` **(Selected)** | `90.0%` | `0.0%` | `60.0%` |
| `0.78` | `85.0%` | `0.0%` | `55.0%` |
| `0.80` | `80.0%` | `0.0%` | `50.0%` |
| `0.82` | `75.0%` | `0.0%` | `40.0%` |
| `0.85` | `65.0%` | `0.0%` | `35.0%` |

## Why Semantic Similarity Alone Fails

In our empirical benchmark, near-miss queries (such as changing `September 4` to `September 5` or swapping customer IDs) scored an average similarity of `0.7927`, with some reaching as high as `0.985`.

At the calibrated retrieval threshold of `0.75`, **60.0% of near-miss queries** pass the semantic similarity test because they share almost identical vocabulary. A naive cache relying on embeddings alone would return invalid, dangerous results (e.g. flight restrictions for the wrong day).

Axiom prevents this by gatekeeping every semantic candidate with strict canonical argument compatibility.
