import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

df = pd.read_csv("../DeepEnzyme/Data/Fig/PafA.tsv", delimiter='\t')
df2 = pd.read_csv("../DeepEnzyme/Data/Fig/science.abf8761_Data-S1.csv")

for i in range(len(df2['variant'])):
    if df2['variant'][i] == 'WT':
        wt = df2['kcat_cMUP_s-1'][i]

x_exp, y_exp = [], []
for i in range(len(df['Predicted kcat'])):
    if df['experiment kcat'][i] < wt:
        x_exp.append(df['experiment kcat'][i])
    if df['experiment kcat'][i] > wt:
        y_exp.append(df['experiment kcat'][i])

t, p = stats.ttest_ind(x_exp, y_exp)
p = round(p, 4)

colors = ['#5F9E98', '#CD885D']
sns.violinplot(data=[x_exp, y_exp], inner="box", palette=colors, width=0.5)

x1, x2 = 0, 1
y, h, col = 350, 0.1, 'k'
plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
plt.text((x1 + x2) * .5, y + h, f"P value={p}", ha='center', va='bottom', color=col)
labels = ['low $k$$_\mathregular{cat}$ mutantions', 'high $k$$_\mathregular{cat}$ mutantions']
x = np.arange(0, 2)
plt.title("PafA")
plt.xticks(x, labels)
plt.ylabel('Experimental $k$$_\mathregular{cat}$ Value', fontsize=12)

plt.show()
#plt.savefig("../../figure/Fig4d.pdf", dpi=600, bbox_inches='tight')

