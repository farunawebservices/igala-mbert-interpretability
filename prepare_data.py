import pandas as pd

print("📊 Preparing data for interpretability analysis...\n")

# Use the parallel corpus (224 sentences with English translations)
df = pd.read_csv('data/igala_english_parallel.csv')

print(f"Total sentence pairs: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Show first row to verify structure
print(f"\nFirst sentence pair:")
print(df.head(1))

# Select 50 diverse examples
analysis_df = df.sample(n=min(50, len(df)), random_state=42).reset_index(drop=True)

# Save
analysis_df.to_csv('data/igala_probe_sentences.csv', index=False)

print(f"\n✅ Created analysis set: {len(analysis_df)} sentence pairs")
print("\nSample:")
print(analysis_df.head(3))
