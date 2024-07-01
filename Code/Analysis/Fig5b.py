import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

df = pd.read_csv("../../../DeepEnzyme/Data/Fig/PafA_att.csv")

x = df['site'].values.tolist()
y = df['binding/active_site'].values.tolist()
y = y[0:11]
x_max = max(x)
x_min = min(x)
xx = []
yy = []
for i in x:
    xx.append((i - x_min) / (x_max - x_min))

for i in y:
    yy.append((i - x_min) / (x_max - x_min))

fig, ax = plt.subplots(figsize=(4, 6))
plt.rcParams.update({'font.size': 12})

bp = ax.boxplot([xx, yy], vert=True, patch_artist=True, labels=['General site', 'Binding/Active site'],
                medianprops={'color': 'black'})

colors = ['#7EBCAE', '#E79C88']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

t, p = stats.ttest_ind(xx, yy)
p = round(p, 4)

ax.text(0.5, 0.9, f'P value = {p}', transform=ax.transAxes, ha='center', va='center', fontsize=12)

ax.set_ylabel('Weight of Position', fontsize=12)

plt.tick_params(axis='both', labelsize=12)
plt.tight_layout()

plt.show()
#plt.savefig("../../../DeepEnzyme/Results/Figures/Fig5b.pdf", dpi=600, bbox_inches='tight')
