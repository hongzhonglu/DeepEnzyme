import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

df = pd.read_excel("../../../DeepEnzyme/Data/Fig/mmc3.xlsx")

nonsense = []
missense = []

for i in range(len(df['class'])):
    if df['class'][i] == 'nonsense' and np.isnan(df['activity_score'][i]) == False:
        nonsense.append(df['activity_score'][i])
    if df['class'][i] == 'missense' and np.isnan(df['activity_score'][i]) == False:
        missense.append(df['activity_score'][i])

x = nonsense
y = missense

t1, p1 = stats.ttest_ind(x, y)
p1 = round(p1, 4)

colors = ['#5F9E98', '#CD885D']

plots = sns.violinplot(data=[x, y], inner="box", palette=colors, width=0.5)

x1, x2 = 0, 1
y1, y2, h1, h2, col = 3, 3, 0.2, 0.2, 'k'
plt.plot([x1, x1, x2, x2], [y1, y1 + h1, y1 + h1, y1], lw=1.5, c=col)
plt.text((x1 + x2) * .5, y1 + h1, f"P value={p1}", ha='center', va='bottom', color=col)
labels = ['nonsense variants', 'missense variants']
x = np.arange(0, 2)
plt.title("CYP2C9")
plt.xticks(x, labels)
plt.ylabel('Experimental activity score', fontsize=12)

plt.show()
#plt.savefig("../../../DeepEnzyme/Results/Figures/Fig4b.pdf", dpi=600, bbox_inches='tight')
