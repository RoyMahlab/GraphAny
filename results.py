import numpy as np



# trained on Pubmed
gossipcop = np.array([50.22999954223633, 43.84000015258789, 52.04999923706055])
mutag = np.array([56.40999984741211, 61.540000915527344, 56.40999984741211])
nci109 = np.array([51.45000076293945, 52.060001373291016, 51.09000015258789])
nci1 = np.array([48.65999984741211, 49.63999938964844, 49.150001525878906])
proteins = np.array([56.25, 61.15999984741211, 62.95000076293945])

print(f"gossipcop: {gossipcop.mean():.2f} +- {gossipcop.std():.2f}")
print(f"mutag: {mutag.mean():.2f} +- {mutag.std():.2f}")
print(f"nci109: {nci109.mean():.2f} +- {nci109.std():.2f}")
print(f"nci1: {nci1.mean():.2f} +- {nci1.std():.2f}")
print(f"proteins: {proteins.mean():.2f} +- {proteins.std():.2f}")