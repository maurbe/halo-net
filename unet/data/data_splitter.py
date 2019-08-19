"""
    Script to split the entire data ID range into train and validation set.
    This enables to train/validate only on a subset if desired.
"""
import numpy as np

train_ids  = np.arange(0, 512, 1)
test_ids   = np.arange(0, 512, 1)

np.savez('ids.npz', train_ids=train_ids, test_ids=test_ids)
print('\nData indexing complete!')
