import numpy as np
import pandas as pd

results = {"Arxiv": [58.59, 58.59, 58.74], 
           "Cora": [80.19, 80.09, 80.0],
           "Pubmed": [77.09, 77.09, 76.90],
           "Wiki": [58.08, 57.50, 57.79],
           "CoPhysics": [92.38, 93.05, 92.44]}
new_dict = {}
print("Results for GraphAny on PubMed:")

for dataset, scores in results.items():
    mean_score = round(np.mean(scores),2)
    std_score = round(np.std(scores),2)
    string = f"{mean_score:.2f} ± {std_score:.2f}"
    print(string)
    new_dict[dataset] = (dataset, mean_score, std_score)
df = pd.DataFrame.from_dict(new_dict, orient="index", columns=["Dataset", "Mean", "Std"])
print("Note: The results are averaged over 3 runs.")
df.to_csv("graphany_pubmed_results.csv", index=False)

