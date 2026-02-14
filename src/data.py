import pandas as pd

# Load CSV
df = pd.read_csv('gnomAD_HIST1H4A.csv')

# Normalize column names
df.columns = df.columns.str.strip().str.lower()
print("Normalized Columns:", df.columns.tolist())

# Fill missing values
df.fillna(0, inplace=True)

# Use safe key access
df['label'] = df.apply(lambda row: 1 if (row.get('af', 0) > 0.05 and row.get('ac_hom', 0) > 5) else 0, axis=1)

# Save labeled dataset
df.to_csv('gnomad_labeled.csv', index=False)

print("✅ Labeled dataset saved as 'gnomad_labeled.csv'")
print(df[['af', 'ac_hom', 'label']].head(10))
