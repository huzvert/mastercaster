import pandas as pd
import numpy as np

print("=" * 70)
print("EXTENDING DATASET: May-Dec 2024 → May-Dec 2025")
print("=" * 70)

# Load original data
df = pd.read_csv('../data/processed/merged_data_clean.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"\nOriginal data: {df['date'].min()} to {df['date'].max()}")
print(f"Total rows: {len(df)}")

# Get May-Dec 2023 and 2024 data
may_dec_2023 = df[(df['date'] >= '2023-05-02') & (df['date'] <= '2023-12-31')].copy()
may_dec_2024 = df[(df['date'] >= '2024-05-02') & (df['date'] <= '2024-12-31')].copy()

print(f"\nMay-Dec 2023: {len(may_dec_2023)} rows")
print(f"May-Dec 2024: {len(may_dec_2024)} rows")

# Average the two years
may_dec_2025 = may_dec_2024.copy()

# Average all numeric columns
numeric_cols = may_dec_2025.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    # Handle cases where 2023 and 2024 have different lengths
    min_len = min(len(may_dec_2023), len(may_dec_2024))
    may_dec_2025[col].iloc[:min_len] = (
        may_dec_2023[col].iloc[:min_len].values + 
        may_dec_2024[col].iloc[:min_len].values
    ) / 2

# Shift dates to 2025
may_dec_2025['date'] = may_dec_2025['date'] + pd.DateOffset(years=1)

print(f"\nMay-Dec 2025 created: {may_dec_2025['date'].min()} to {may_dec_2025['date'].max()}")
print(f"May-Dec 2025 rows: {len(may_dec_2025)}")

# Combine: Original (until May 1, 2025) + New (May 2 - Dec 31, 2025)
df_extended = pd.concat([df, may_dec_2025], ignore_index=True)
df_extended = df_extended.sort_values('date').reset_index(drop=True)

print(f"\n✅ Extended dataset: {df_extended['date'].min()} to {df_extended['date'].max()}")
print(f"Total rows: {len(df_extended)}")

# Save
output_path = '../data/processed/merged_data_extended.csv'
df_extended.to_csv(output_path, index=False)

print(f"\n✅ Saved to: {output_path}")
print("\n" + "=" * 70)
print("✅ FULL EXTENDED DATASET CREATED!")
print("=" * 70)
print(f"\nℹ️  You now have data from Jan 1, 2023 to Dec 31, 2025")
print(f"ℹ️  Gap filled: May 2 - Dec 31, 2025 ({len(may_dec_2025)} days)")