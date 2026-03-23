import numpy as np

def expand_features(features):
    features = np.array(features)

    if len(features) != 24:
        raise ValueError("Expected 24 input features")

    mean = np.mean(features)
    std = np.std(features)
    max_val = np.max(features)
    min_val = np.min(features)

    extra = []

    # 19 * 4 = 76 → total = 100
    for _ in range(19):
        extra.extend([mean, std, max_val, min_val])

    expanded = np.concatenate([features, extra])

    
    if len(expanded) < 102:
        expanded = np.pad(expanded, (0, 102 - len(expanded)), mode='constant')

    return expanded[:102].reshape(1, -1)