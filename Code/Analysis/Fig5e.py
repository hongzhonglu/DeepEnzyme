import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

df = pd.read_csv("../../../DeepEnzyme/Data/Fig/P00558_att.csv")
with open('../../../DeepEnzyme/Data/Fig/P00558_att.txt', 'r') as input:
    att = input.readlines()

x = df['site'].values.tolist()
y = df['binding/active_site'].values.tolist()
y = y[0:17]

fig, ax = plt.subplots(figsize=(4, 6))

plt.rcParams.update({'font.size': 12})

bp = ax.boxplot([x, y], vert=True, patch_artist=True, labels=['General site', 'Binding/Active site'],
                medianprops={'color': 'black'})

colors = ['#5F9E98', '#CD885D']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

t, p = stats.ttest_ind(x, y)
p = round(p, 4)

ax.text(0.5, 0.9, f'P value = {p}', transform=ax.transAxes, ha='center', va='center', fontsize=12)
ax.set_ylabel('Weight of Position', fontsize=12)

plt.tick_params(axis='both', labelsize=12)
plt.tight_layout()

plt.show()
#plt.savefig("../../../DeepEnzyme/Results/Figures/Fig5e.pdf", dpi=600, bbox_inches='tight')
