import pandas as pd
import numpy as np

# Load the CSV
df = pd.read_csv("graphany/cora.csv")

# Compute row-wise mean and std (excluding first column)
numeric_data = df.iloc[:, 1:].apply(
    pd.to_numeric, errors="coerce"
)  # Convert to numeric in case of strings

means = numeric_data.mean(axis=1)
stds = numeric_data.std(axis=1)

# Create new column with "mean ± std" format
df["mean ± std"] = means.round(2).astype(str) + " ± " + stds.round(2).astype(str)

# Save the modified CSV
df.to_csv("graphany/output.csv", index=False)
