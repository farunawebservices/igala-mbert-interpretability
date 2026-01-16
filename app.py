import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Igala Interpretability Explorer", layout="wide")

# Load attention data
@st.cache_resource
def load_data():
    with open('outputs/attention_patterns.pkl', 'rb') as f:
        return pickle.load(f)

data = load_data()

st.title("🔬 Mechanistic Interpretability: mBERT Attention Analysis for Igala")
st.markdown("**Analyzing how multilingual BERT processes Igala vs. English**")

# Sidebar
st.sidebar.header("⚙️ Settings")
pair_idx = st.sidebar.slider("Select sentence pair", 0, len(data)-1, 0)
layer_num = st.sidebar.slider("Select layer (0-11)", 0, 11, 6)

# Get selected data
pair = data[pair_idx]
igala_data = pair['igala']
english_data = pair['english']

# Display sentences
col1, col2 = st.columns(2)

with col1:
    st.subheader("🇳🇬 Igala")
    st.info(igala_data['text'])
    st.caption(f"Tokens: {igala_data['num_tokens']}")

with col2:
    st.subheader("🇬🇧 English")
    st.info(english_data['text'])
    st.caption(f"Tokens: {english_data['num_tokens']}")

# Attention heatmaps
st.markdown("---")
st.subheader(f"🔥 Attention Heatmap - Layer {layer_num}")

col1, col2 = st.columns(2)

def create_attention_heatmap(attention_matrix, tokens, title):
    """Create plotly heatmap for attention"""
    fig = go.Figure(data=go.Heatmap(
        z=attention_matrix,
        x=tokens,
        y=tokens,
        colorscale='Blues',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Target Token",
        yaxis_title="Source Token",
        height=500,
        xaxis={'side': 'bottom'},
        yaxis={'autorange': 'reversed'}
    )
    
    return fig

with col1:
    igala_att = igala_data['layer_attentions'][layer_num]
    fig_igala = create_attention_heatmap(igala_att, igala_data['tokens'], "Igala Attention")
    st.plotly_chart(fig_igala, use_container_width=True)

with col2:
    english_att = english_data['layer_attentions'][layer_num]
    fig_english = create_attention_heatmap(english_att, english_data['tokens'], "English Attention")
    st.plotly_chart(fig_english, use_container_width=True)

# Statistics comparison
st.markdown("---")
st.subheader("📊 Attention Pattern Statistics")

col1, col2, col3 = st.columns(3)

# Calculate statistics for this layer
igala_entropy = -np.sum(igala_att * np.log(igala_att + 1e-10), axis=-1).mean()
english_entropy = -np.sum(english_att * np.log(english_att + 1e-10), axis=-1).mean()

igala_max_att = igala_att.max()
english_max_att = english_att.max()

with col1:
    st.metric("Igala Attention Entropy", f"{igala_entropy:.3f}")
    st.metric("English Attention Entropy", f"{english_entropy:.3f}")
    
with col2:
    st.metric("Igala Max Attention", f"{igala_max_att:.3f}")
    st.metric("English Max Attention", f"{english_max_att:.3f}")

with col3:
    st.metric("Entropy Difference", f"{abs(igala_entropy - english_entropy):.3f}")
    st.metric("Max Attention Difference", f"{abs(igala_max_att - english_max_att):.3f}")

# Layer-wise comparison
st.markdown("---")
st.subheader("📈 Cross-Layer Analysis")

# Compute average attention per layer
igala_layer_stats = []
english_layer_stats = []

for layer in range(12):
    igala_layer_att = igala_data['layer_attentions'][layer]
    english_layer_att = english_data['layer_attentions'][layer]
    
    igala_layer_stats.append({
        'layer': layer,
        'mean_attention': igala_layer_att.mean(),
        'max_attention': igala_layer_att.max(),
        'entropy': -np.sum(igala_layer_att * np.log(igala_layer_att + 1e-10), axis=-1).mean()
    })
    
    english_layer_stats.append({
        'layer': layer,
        'mean_attention': english_layer_att.mean(),
        'max_attention': english_layer_att.max(),
        'entropy': -np.sum(english_layer_att * np.log(english_layer_att + 1e-10), axis=-1).mean()
    })

df_igala = pd.DataFrame(igala_layer_stats)
df_english = pd.DataFrame(english_layer_stats)

# Plot comparison
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_igala['layer'], y=df_igala['entropy'], 
                         mode='lines+markers', name='Igala', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df_english['layer'], y=df_english['entropy'], 
                         mode='lines+markers', name='English', line=dict(color='red')))

fig.update_layout(
    title="Attention Entropy Across Layers",
    xaxis_title="Layer",
    yaxis_title="Entropy",
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Built by Godwin Faruna Abuh** | [Portfolio](https://your-portfolio.com) | [HuggingFace](https://huggingface.co/Faruna01)")
