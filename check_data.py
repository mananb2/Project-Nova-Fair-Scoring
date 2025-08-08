import pandas as pd

# Load your generated dataset
df = pd.read_csv("data/partners_synthetic.csv")

print("Shape:", df.shape)
print("\nMissing values per column:\n", df.isna().sum())

# Quick ranges for key numeric columns
print("\nSummary stats:\n", df[['avg_rating','on_time_pct','cancel_rate','weekly_earnings']].describe())

# Approval rate snapshot (650 threshold)
df['approved'] = (df['nova_score'] >= 650).astype(int)
print("\nOverall approval rate:", round(df['approved'].mean(), 3))
print("Approval rate by city_tier:\n", df.groupby('city_tier')['approved'].mean().round(3))
