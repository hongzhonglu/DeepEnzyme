import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("../../../DeepEnzyme/Data/Fig/PafA_att.csv")
with open('../../../DeepEnzyme/Data/Fig/PafA_att.txt', 'r') as input:
    att = input.readlines()

x = df['site'].values.tolist()
y = df['binding/active_site'].values.tolist()
y = y[0:11]
data = []
x_max = max(x)
x_min = min(x)
for i in att:
    data.append((float(i.split('\n')[0]) - x_min) / (x_max - x_min))

x = np.arange(len(data))

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(x, data, '--', color='#7EBCAE', linewidth=2, markersize=4)

# sites from uniprot
pos = [38, 79, 100, 162, 163, 164, 305, 309, 352, 353, 486]
for i in pos:
    plt.plot(x[i], data[i], marker='o', markersize=8, color='#E79C88')
plt.scatter(x[38], data[38], marker='o', label='Binding/Active site', color='#E79C88')

plt.xlabel('Residue Position', fontsize=12)
plt.ylabel('Weight of Position', fontsize=12)

plt.tick_params(axis='both', labelsize=12)
plt.legend(loc='upper right', fontsize=12)

plt.tight_layout()

plt.show()
#plt.savefig("../../../DeepEnzyme/Results/Figures/Fig5a.pdf", dpi=600, bbox_inches='tight')
