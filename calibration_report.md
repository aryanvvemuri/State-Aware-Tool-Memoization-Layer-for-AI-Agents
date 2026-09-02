# Axiom Semantic Threshold Calibration Report

- **Embedding Model:** `all-MiniLM-L6-v2`
- **Calibrated Default Threshold:** `0.82`
- **Equivalent Pairs Mean Similarity:** `0.7115`
- **Near-Miss Pairs Mean Similarity:** `0.8077`
- **Unrelated Pairs Mean Similarity:** `0.0157`

## Why Semantic Similarity Alone Fails

In our benchmark, near-miss queries (such as changing `September 4` to `September 5`) scored an average similarity of `0.8077`. If an agent used semantic similarity alone, it would return stale or incorrect data for a different date or different flight.

Axiom prevents this by gatekeeping semantic candidates with strict canonical argument compatibility.
