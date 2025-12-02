import pandas as pd

# Load both datasets (just file names, not dataset/Fake.csv)
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

# Add labels
fake_df['label'] = 'FAKE'
true_df['label'] = 'REAL'

# Combine
combined_df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
combined_df.to_csv('news.csv', index=False)

print("✅ Combined dataset saved as news.csv")
