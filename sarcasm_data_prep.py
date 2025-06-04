import pandas as pd
from sklearn.model_selection import train_test_split

# Load the new dataset
data = pd.read_csv('Sarcasm_Headlines_Dataset.csv')

# Check dataset shape and distribution
print(f"Dataset shape: {data.shape}")
print(f"Label distribution:\n{data['is_sarcastic'].value_counts()}")

# Separate positive (sarcastic) and negative (non-sarcastic) samples
data_pos = data[data['is_sarcastic'] == 1]  # Sarcastic headlines: 11,724
data_neg = data[data['is_sarcastic'] == 0]  # Non-sarcastic headlines: 14,985

print(f"Sarcastic samples: {len(data_pos)}")
print(f"Non-sarcastic samples: {len(data_neg)}")

# Calculate split sizes based on available data
# Using approximately 70% train, 15% validation, 15% test split

# For sarcastic samples (11,724 total)
pos_train_size = int(0.7 * len(data_pos))      # ~8,207
pos_val_size = int(0.15 * len(data_pos))       # ~1,759
pos_test_size = len(data_pos) - pos_train_size - pos_val_size  # ~1,758

# For non-sarcastic samples (14,985 total)
neg_train_size = int(0.7 * len(data_neg))      # ~10,490
neg_val_size = int(0.15 * len(data_neg))       # ~2,248
neg_test_size = len(data_neg) - neg_train_size - neg_val_size  # ~2,247

print(f"\nSplit sizes:")
print(f"Positive - Train: {pos_train_size}, Val: {pos_val_size}, Test: {pos_test_size}")
print(f"Negative - Train: {neg_train_size}, Val: {neg_val_size}, Test: {neg_test_size}")

# Create splits for positive samples
data_pos_tr = data_pos.iloc[:pos_train_size]
data_pos_cv = data_pos.iloc[pos_train_size:pos_train_size + pos_val_size]
data_pos_test = data_pos.iloc[pos_train_size + pos_val_size:]

# Create splits for negative samples
data_neg_tr = data_neg.iloc[:neg_train_size]
data_neg_cv = data_neg.iloc[neg_train_size:neg_train_size + neg_val_size]
data_neg_test = data_neg.iloc[neg_train_size + neg_val_size:]

# Combine positive and negative samples for each split
data_tr = pd.concat([data_pos_tr, data_neg_tr], ignore_index=True)
data_cv = pd.concat([data_pos_cv, data_neg_cv], ignore_index=True)
data_test = pd.concat([data_pos_test, data_neg_test], ignore_index=True)

# Shuffle the combined datasets
data_tr = data_tr.sample(frac=1, random_state=42).reset_index(drop=True)
data_cv = data_cv.sample(frac=1, random_state=42).reset_index(drop=True)
data_test = data_test.sample(frac=1, random_state=42).reset_index(drop=True)

# Print final split information
print(f"\nFinal dataset splits:")
print(f"Training set: {len(data_tr)} samples")
print(f"  - Sarcastic: {sum(data_tr['is_sarcastic'] == 1)}")
print(f"  - Non-sarcastic: {sum(data_tr['is_sarcastic'] == 0)}")

print(f"Validation set: {len(data_cv)} samples")
print(f"  - Sarcastic: {sum(data_cv['is_sarcastic'] == 1)}")
print(f"  - Non-sarcastic: {sum(data_cv['is_sarcastic'] == 0)}")

print(f"Test set: {len(data_test)} samples")
print(f"  - Sarcastic: {sum(data_test['is_sarcastic'] == 1)}")
print(f"  - Non-sarcastic: {sum(data_test['is_sarcastic'] == 0)}")

# Save the splits (optional)
data_tr.to_csv('data/processed/train.csv', index=False)
data_cv.to_csv('data/processed/validation.csv', index=False)
data_test.to_csv('data/processed/test.csv', index=False)