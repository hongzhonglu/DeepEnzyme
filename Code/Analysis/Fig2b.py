import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, gaussian_kde
from sklearn.metrics import mean_squared_error

df = pd.read_csv('../../../DeepEnzyme/Data/Fig/Fig2.csv')
predicted_logkcat = []
experiment_logkcat = []

for i in df.index:
    if df['label'][i] == 1:
        predicted_logkcat.append(df['Predicted logkcat'][i])
        experiment_logkcat.append(df['experiment logkcat'][i])

pearson_corr, p_value = pearsonr(experiment_logkcat, predicted_logkcat)
rmse = np.sqrt(mean_squared_error(experiment_logkcat, predicted_logkcat))

plt.figure(figsize=(1.5, 1.5))
plt.axes([0.12, 0.12, 0.83, 0.83])

plt.tick_params(direction='in')
plt.tick_params(which='major', length=1.5)
plt.tick_params(which='major', width=0.4)

kcat_values_vstack = np.vstack([experiment_logkcat, predicted_logkcat])
experimental_predicted = gaussian_kde(kcat_values_vstack)(kcat_values_vstack)

ax = plt.scatter(x=experiment_logkcat, y=predicted_logkcat, c=experimental_predicted, s=3, edgecolor=[])

cbar = plt.colorbar(ax)
cbar.ax.tick_params(labelsize=4)
cbar.set_label('Density', size=5)

plt.text(-5, 5, 'PCC = %.2f' % pearson_corr, fontweight="normal", fontsize=4)
plt.text(-5, 4.5, 'P value = 0', fontweight="normal", fontsize=4)
plt.text(3, -2.5, 'Wlid-type', fontweight="normal", fontsize=4)

plt.xlabel("Experimental $k$$_\mathregular{cat}$ value", fontdict={'weight': 'normal', 'size': 5}, fontsize=5)
plt.ylabel('Predicted $k$$_\mathregular{cat}$ value', fontdict={'weight': 'normal', 'size': 5}, fontsize=5)

plt.xlim([-6, 7])
plt.xlim([-6, 7])

plt.xticks(fontsize=4)
plt.yticks(fontsize=4)

ax = plt.gca()
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['left'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)
ax.spines['right'].set_linewidth(0.5)

plt.show()
# plt.savefig("../../../DeepEnzyme/Results/Figures/Fig2b.pdf", dpi=600, bbox_inches='tight')
