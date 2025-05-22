import pandas as pd
import numpy as np
from math import log2

# Sample training dataset (from image)
data = [
    ['<=30', 'high', 'no', 'fair', 'no'],
    ['<=30', 'high', 'no', 'excellent', 'no'],
    ['31...40', 'high', 'no', 'fair', 'yes'],
    ['>40', 'medium', 'no', 'fair', 'yes'],
    ['>40', 'low', 'yes', 'fair', 'yes'],
    ['>40', 'low', 'yes', 'excellent', 'no'],
    ['31...40', 'low', 'yes', 'excellent', 'yes'],
    ['<=30', 'medium', 'no', 'fair', 'no'],
    ['<=30', 'low', 'yes', 'fair', 'yes'],
    ['>40', 'medium', 'yes', 'fair', 'yes'],
    ['<=30', 'medium', 'yes', 'excellent', 'yes'],
    ['31...40', 'medium', 'no', 'excellent', 'yes'],
    ['31...40', 'high', 'yes', 'fair', 'yes'],
    ['>40', 'medium', 'no', 'excellent', 'no']
]

# Load into a DataFrame
df = pd.DataFrame(data, columns=['age', 'income', 'student', 'credit_rating', 'buys_computer'])

# Function to calculate entropy
def entropy(target_col):
    values, counts = np.unique(target_col, return_counts=True)
    entropy = 0
    for i in range(len(values)):
        prob = counts[i] / np.sum(counts)
        entropy -= prob * log2(prob)
    return entropy

# Function to calculate information gain
def info_gain(data, split_attr, target_attr):
    total_entropy = entropy(data[target_attr])
    values, counts = np.unique(data[split_attr], return_counts=True)
    
    weighted_entropy = 0
    for i in range(len(values)):
        subset = data[data[split_attr] == values[i]]
        weighted_entropy += (counts[i] / np.sum(counts)) * entropy(subset[target_attr])
        
    return total_entropy - weighted_entropy

# ID3 algorithm (simplified)
def id3(data, original_data, features, target_attr, parent_node_class=None):
    # If all target values have same class, return that class
    if len(np.unique(data[target_attr])) <= 1:
        return np.unique(data[target_attr])[0]
    
    # If dataset is empty, return majority class of original data
    elif len(data) == 0:
        return np.unique(original_data[target_attr])[np.argmax(np.unique(original_data[target_attr], return_counts=True)[1])]
    
    # If no features left, return majority class
    elif len(features) == 0:
        return parent_node_class
    
    # Get majority class of current node
    else:
        parent_node_class = np.unique(data[target_attr])[np.argmax(np.unique(data[target_attr], return_counts=True)[1])]
        
        # Select feature with max information gain
        gains = [info_gain(data, feature, target_attr) for feature in features]
        best_feature_index = np.argmax(gains)
        best_feature = features[best_feature_index]
        
        # Create tree structure
        tree = {best_feature: {}}
        features = [f for f in features if f != best_feature]
        
        for value in np.unique(data[best_feature]):
            subset = data[data[best_feature] == value]
            subtree = id3(subset, original_data, features, target_attr, parent_node_class)
            tree[best_feature][value] = subtree
        
        return tree

# Run the algorithm
features = ['age', 'income', 'student', 'credit_rating']
tree = id3(df, df, features, 'buys_computer')
print("Decision Tree:")
print(tree)
