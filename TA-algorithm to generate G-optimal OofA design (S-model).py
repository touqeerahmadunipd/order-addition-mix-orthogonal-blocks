#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from itertools import combinations
from numpy.linalg import matrix_rank, det
import random


block1 = np.array([
    [0.168, 0.832, 0.000,  1,  0,  0],
    [0.168, 0.832, 0.000, -1,  0,  0],
    [0.832, 0.000, 0.168,  0, -1,  0],
    [0.832, 0.000, 0.168,  0,  1,  0],
    [0.000, 0.168, 0.832,  0,  0,  1],
    [0.000, 0.168, 0.832,  0,  0, -1],
    [0.333, 0.333, 0.334,  1,  1,  1],
    [0.333, 0.333, 0.334,  1,  1, -1],
    [0.333, 0.333, 0.334,  1, -1, -1],
    [0.333, 0.333, 0.334, -1,  1,  1],
    [0.333, 0.333, 0.334, -1, -1,  1],
    [0.333, 0.333, 0.334, -1, -1, -1],
])

block2 = np.array([
    [0.168, 0.000, 0.832,  0,  1,  0],
    [0.168, 0.000, 0.832,  0, -1,  0],
    [0.832, 0.168, 0.000, -1,  0,  0],
    [0.832, 0.168, 0.000,  1,  0,  0],
    [0.000, 0.832, 0.168,  0,  0, -1],
    [0.000, 0.832, 0.168,  0,  0,  1],
    [0.333, 0.333, 0.334,  1,  1,  1],
    [0.333, 0.333, 0.334,  1,  1, -1],
    [0.333, 0.333, 0.334,  1, -1, -1],
    [0.333, 0.333, 0.334, -1,  1,  1],
    [0.333, 0.333, 0.334, -1, -1,  1],
    [0.333, 0.333, 0.334, -1, -1, -1],
])

# Generate Model Matrix
def generate_model_matrix(design):
    x1, x2, x3, z12, z13, z23 = design.T
    return np.column_stack([
        x1, x2, x3, z12, z13, z23,
        x1*x2, x1*x3, x2*x3,
        x1*z12, x3*z13, x2*z23
    ])

# Orthogonal blocking conditions for fitting S-model
def check_strict_orthogonality(b1, b2):
    for i in range(6):
        if not np.isclose(np.sum(b1[:, i]), np.sum(b2[:, i]), atol=1e-3):
            return False
    interactions = [(0, 1), (0, 2), (1, 2), (0, 3), (2, 4), (1, 5)]
    for i, j in interactions:
        if not np.isclose(np.sum(b1[:, i] * b1[:, j]), np.sum(b2[:, i] * b2[:, j]), atol=1e-3):
            return False
    return True

# Running G-efficiency optimization
def optimize_strict_g_efficiency(b1_all, b2_all, p=12, k=8, max_iter=100000):
    best_eff = -np.inf
    best_blocks = None

    for _ in range(max_iter):
        idx1 = sorted(random.sample(range(p), k))
        idx2 = sorted(random.sample(range(p), k))
        b1 = b1_all[idx1]
        b2 = b2_all[idx2]

        if not check_strict_orthogonality(b1, b2):
            continue

        design = np.vstack([b1, b2])
        X = generate_model_matrix(design)
        if matrix_rank(X) < X.shape[1]:
            continue

        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            continue  

        max_pred_var = max(np.dot(row, XtX_inv @ row) for row in X)
        g_eff = (X.shape[1] / max_pred_var) * 100

        if g_eff > best_eff:
            best_eff = g_eff
            best_blocks = (b1, b2)

    return best_blocks, best_eff


strict_blocks, strict_g_eff = optimize_strict_g_efficiency(block1, block2)

# output
if strict_blocks:
    b1_df = pd.DataFrame(strict_blocks[0], columns=["x1", "x2", "x3", "z12", "z13", "z23"])
    b2_df = pd.DataFrame(strict_blocks[1], columns=["x1", "x2", "x3", "z12", "z13", "z23"])
    b1_df["Block"] = "Block 1"
    b2_df["Block"] = "Block 2"
    final_df = pd.concat([b1_df, b2_df], ignore_index=True)
else:
    final_df = pd.DataFrame({"Error": ["No valid design found under strict conditions."]})

print(final_df)


# In[ ]:




