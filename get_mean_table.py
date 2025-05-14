import pandas as pd


df = pd.read_csv("results.csv")
# Compute mean and std
means = df.iloc[:, 1:].mean(axis=1)
stds = df.iloc[:, 1:].std(axis=1)

# Create a new column with "mean ± std" format
df['mean_std'] = means.round(2).astype(str) + ' ± ' + stds.round(2).astype(str)
df.to_csv("results_mean_std.csv", index=False)
