import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import pickle

print("🔍 Extracting attention patterns from mBERT...\n")

# Load mBERT
model_name = "bert-base-multilingual-cased"
print(f"📥 Loading {model_name}...")

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_attentions=True)

print("✅ Model loaded!\n")

def analyze_sentence(text, language='igala'):
    """Extract attention patterns for a sentence"""
    
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    
    # Get outputs with attention
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract attention (12 layers, 12 heads each)
    attentions = outputs.attentions  # Tuple of 12 tensors
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Average attention across heads for each layer
    layer_attentions = []
    for layer_att in attentions:
        # layer_att shape: (1, 12_heads, seq_len, seq_len)
        avg_att = layer_att[0].mean(dim=0).numpy()  # Average across 12 heads
        layer_attentions.append(avg_att)
    
    return {
        'text': text,
        'language': language,
        'tokens': tokens,
        'num_tokens': len(tokens),
        'layer_attentions': layer_attentions  # List of 12 numpy arrays
    }

# Load probe sentences
df = pd.read_csv('data/igala_probe_sentences.csv')

print(f"Processing {len(df)} sentence pairs...\n")

results = []

for idx, row in df.iterrows():
    # Analyze Igala
    igala_data = analyze_sentence(row['igala'], 'igala')
    
    # Analyze English
    english_data = analyze_sentence(row['english'], 'english')
    
    results.append({
        'pair_id': idx,
        'igala': igala_data,
        'english': english_data
    })
    
    if (idx + 1) % 10 == 0:
        print(f"✅ Processed {idx + 1}/{len(df)} pairs")

print(f"\n✅ Extraction complete!")

# Save as pickle (easier for numpy arrays)
with open('outputs/attention_patterns.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"💾 Saved to outputs/attention_patterns.pkl")

# Quick stats
igala_avg_len = np.mean([r['igala']['num_tokens'] for r in results])
english_avg_len = np.mean([r['english']['num_tokens'] for r in results])

print(f"\n📊 Statistics:")
print(f"   Igala avg tokens: {igala_avg_len:.1f}")
print(f"   English avg tokens: {english_avg_len:.1f}")
