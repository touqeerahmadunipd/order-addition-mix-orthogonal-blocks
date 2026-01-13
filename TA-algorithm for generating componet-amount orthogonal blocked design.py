#!/usr/bin/env python
# coding: utf-8

# In[6]:


# TA-algorithm to generate G-optimal OofA mixture/component-amount orthogonal block design

import numpy as np
import pandas as pd
import random
from numpy.linalg import matrix_rank

# Define model matrix generator
def generate_model_matrix(design):
    a1, a2, a3, z12, z13, z23 = design.T
    return np.column_stack([
        np.ones_like(a1),     
        a1, a2, a3,           
        a1**2, a2**2, a3**2,  
        z12, z13, z23,        
        a1*a2, a1*a3, a2*a3,  
        
    ])

# Define design matrix
block1 = np.array([
    [0, 0, 24,  0,  0,  0],
    [0, 76, 0, 0,  0,  0],
    [24, 0, 76,  0, 1,  0],
    [24, 0, 76,  0,  -1,  0],
    [76, 24, 0,  -1,  0,  0],
    [76, 24, 0,  1,  0, 0],
    [0, 0, 24,  0,  0,  0],
    [0, 24, 76,  0,  0, 1],
    [0, 24, 76,  0,  0, -1],
    [24, 76, 0 ,1,  0,  0],
    [24, 76, 0 ,-1,  0,  0],
    [76, 0, 0, 0, 0, 0],
    [25, 25, 25, 1, 1, 1],
    [25, 25, 25, 1, 1, -1],
    [25, 25, 25, 1, -1, -1],
    [25, 25, 25, -1, -1, -1],
    [25, 25, 25, -1, 1, 1],
    [25, 25, 25, -1, -1, 1],
])

block2 = np.array([
    [0, 24, 0,  0,  0,  0],
    [0, 0, 76,  0, 0,  0],
    [24, 76, 0, 1,  0,  0],
    [24, 76, 0, -1,  0,  0],
    [76, 0, 24,  0,  -1, 0],
    [76, 0, 24,  0,  1, 0],
    [0, 76, 24,  0,  0,  -1],
    [0, 76, 24,  0,  0,  1],
    [0, 0, 76,  0, 0, 0],
    [24, 0, 0, 0,  0,  0],
    [76, 24, 0, -1, 0,  0],
    [76, 24, 0,  1, 0,  0],
    [25, 25, 25, 1, 1, 1],
    [25, 25, 25, 1, 1, -1],
    [25, 25, 25, 1, -1, -1],
    [25, 25, 25, -1, -1, -1],
    [25, 25, 25, -1, 1, 1],
    [25, 25, 25, -1, -1, 1],
])
# Define orthogonality conditions,

def check_relaxed_orthogonality(b1, b2):
    n_cols = b1.shape[1]
    
    
    for i in range(n_cols):
        if not np.isclose(np.sum(b1[:, i]), np.sum(b2[:, i]), atol=1e-3):
            return False

    
    for i in range(3):
        if not np.isclose(np.sum(b1[:, i]**2), np.sum(b2[:, i]**2), atol=1e-3):
            return False
    interactions = [(0, 1), (0, 2), (1, 2)]  # a1a2, a1a3, a2a3
    for i, j in interactions:
        if not np.isclose(np.sum(b1[:, i] * b1[:, j]), np.sum(b2[:, i] * b2[:, j]), atol=1e-3):
            return False

    return True

# Apply the optimization procedure
threshold = 85  
strict_blocks = None
best_eff = -np.inf


for _ in range(2000000):
    b1, b2 = random.sample(list(block1), 10), random.sample(list(block2), 10)
    b1, b2 = np.array(b1), np.array(b2)

    if not check_relaxed_orthogonality(b1, b2):
        continue

    design = np.vstack([b1, b2])
    X = generate_model_matrix(design)
    if matrix_rank(X) < X.shape[1]:
        continue

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        continue

    max_pred_var = max(np.dot(row, XtX_inv @ row) for row in X)
    g_eff = (X.shape[1] / max_pred_var) * 100

    if g_eff > best_eff:
        best_eff = g_eff
        strict_blocks = (b1, b2)
    
    if g_eff >= threshold:
        break

# Print the design
if strict_blocks:
    b1_df = pd.DataFrame(strict_blocks[0], columns=["a1", "a2", "a3", "z12", "z13", "z23"])
    b2_df = pd.DataFrame(strict_blocks[1], columns=["a1", "a2", "a3", "z12", "z13", "z23"])
    b1_df["Block"] = "Block 1"
    b2_df["Block"] = "Block 2"
    final_df = pd.concat([b1_df, b2_df], ignore_index=True)
else:
    final_df = pd.DataFrame({"Error": ["No valid design found meeting the threshold."]})
print(final_df)


# In[ ]:




