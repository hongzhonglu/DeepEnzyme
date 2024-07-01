import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("../../../DeepEnzyme/Data/Fig/P00558_att.csv")
with open('../../../DeepEnzyme/Data/Fig/P00558_att.txt', 'r') as input:
    att = input.readlines()

x = df['site'].values.tolist()
y = df['binding/active_site'].values.tolist()
y = y[0:17]

data = []
for i in att:
    data.append(float(i.split('\n')[0]))

x = np.arange(len(data))

plt.rcParams.update({'font.size': 12})

fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(x, data, '--', color='#5F9E98', linewidth=2, markersize=4)

# sites from uniprot
pos = [24, 25, 26, 39, 63, 64, 65, 66, 123, 171, 220, 313, 344, 373, 374, 375, 376]
for i in pos:
    plt.plot(x[i], data[i], marker='o', markersize=8, color='#CD885D')
plt.scatter(x[24], data[24], marker='o', label='Binding/Active site', color='#CD885D')

plt.xlabel('Residue Position', fontsize=12)
plt.ylabel('Weight of Position', fontsize=12)

plt.tick_params(axis='both', labelsize=12)
plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()

plt.show()
#plt.savefig("../../../DeepEnzyme/Results/Figures/Fig/Fig5d.pdf", dpi=600, bbox_inches='tight')
