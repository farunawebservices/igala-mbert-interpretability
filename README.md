@"
---
title: Igala mBERT Interpretability
emoji: 🔬
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# Mechanistic Interpretability: mBERT Attention Analysis

Explore how multilingual BERT processes Igala vs English through attention pattern analysis.

## Features
- Layer-by-layer attention visualization (12 layers)
- Head-by-head analysis (12 attention heads per layer)
- Side-by-side Igala vs English comparison
- Interactive heatmaps with Plotly
- Pre-computed attention patterns for fast inference
"@ | Out-File -FilePath README.md -Encoding utf8