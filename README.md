# 🔬 Mechanistic Interpretability Analysis

Getting into mBERT's internal mechanisms during Igala-English translation. Visualizes attention patterns, analyzes token alignments, and explores how transformers handle morphologically complex low-resource languages.

## 🎯 Overview

This project investigates **how transformer models actually work** when translating between English and Igala (a low-resource Nigerian language). By visualizing attention heads and analyzing internal representations, we gain insights into:

- How the model aligns tokens across languages
- Which attention heads focus on syntactic vs semantic features
- How morphological complexity affects attention patterns
- Where the model struggles with low-resource data

## 🚀 Live Demo

Explore attention patterns: [https://huggingface.co/spaces/Faruna01/igala-mbert-interpretability](https://huggingface.co/spaces/Faruna01/igala-mbert-interpretability)

## 📊 Features

- ✅ Layer-by-layer attention visualization (12 layers × 12 heads = 144 attention matrices)
- ✅ Token-level alignment heatmaps
- ✅ Interactive Plotly visualizations
- ✅ Comparative analysis across language pairs
- ✅ Morphological feature tracking

## 🛠️ Tech Stack

- **Model**: `bert-base-multilingual-cased` (mBERT)
- **Framework**: PyTorch, TransformerLens
- **Visualization**: Plotly, Matplotlib
- **Frontend**: Streamlit
- **Analysis**: NumPy, Pandas

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/farunawebservices/igala-mbert-interpretability.git
cd igala-mbert-interpretability

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

🔍 Usage
from interpretability import AttentionAnalyzer

# Initialize analyzer
analyzer = AttentionAnalyzer(model_name="bert-base-multilingual-cased")

# Analyze attention patterns
attention_map = analyzer.get_attention(
    source_text="Ọma ẹdu la",  # Igala: "Good morning"
    target_text="Good morning",
    layer=6,
    head=3
)

# Visualize
analyzer.plot_attention_heatmap(attention_map)

📈 Key Findings
Attention Pattern Observations:
Early Layers (0-3): Focus on positional and lexical features

High attention to function words and morphemes

Limited cross-lingual alignment

Middle Layers (4-7): Syntactic structure emerges

Strong subject-verb-object alignment

Attention heads specialize (syntax vs semantics)

Late Layers (8-11): High-level semantic alignment

Cross-lingual token correspondence

Context-dependent disambiguation

Example: "Ọla odu du" → "Good morning"
| Head    | Layer    | Attention Focus                                       |
| ------- | -------- | ----------------------------------------------------- |
| Head 3  | Layer 2  | Morpheme boundaries ("Ọla" splits internally)         |
| Head 7  | Layer 6  | Cross-lingual alignment ("odu" → "morning")           |
| Head 11 | Layer 10 | Contextual meaning ("du" → implicit greeting context) |

⚠️ Limitations
Model Size: mBERT has only 110M parameters; larger models may show different patterns

Dataset: Limited to 3,253 sentence pairs; may not generalize to all Igala contexts

Interpretability Methods: Attention ≠ causation; visualizations show correlation, not mechanism

Language Coverage: Analysis focused on Igala-English; findings may not transfer to other low-resource pairs

Computational: Full 144-head analysis requires significant memory (8GB+ RAM)

🔬 Research Questions Explored
Q1: Do attention patterns differ between high-resource and low-resource translation?

A: Yes - low-resource shows more diffuse attention in early layers

Q2: Which heads specialize for morphological analysis?

A: Heads 2-4 in layers 3-5 show strong morpheme boundary detection

Q3: How does tone marking affect attention?

A: Tone diacritics create distinct attention patterns in layer 6-8

🔮 Future Work
 Compare mBERT vs XLM-R vs mT5 attention patterns

 Causal intervention experiments (patch attention heads)

 Extend to other Niger-Congo languages

 Build attention pattern classifier for error detection

 Investigate multilingual vs monolingual attention differences

📚 References
Vig, J. (2019). A Multiscale Visualization of Attention in the Transformer Model

Clark, K. et al. (2019). What Does BERT Look At?

Ravishankar, V. et al. (2021). Attention vs Non-attention for Low-Resource NMT

📄 License
MIT License - See LICENSE for details

🙏 Acknowledgments
TransformerLens library by Neel Nanda

BertViz visualization tools

Igala language community

📧 Contact
Faruna Godwin Abuh
Applied AI Safety Engineer
📧 farunagodwin01@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/faruna-godwin-abuh-07a22213b/
