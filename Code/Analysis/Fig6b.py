import matplotlib.pyplot as plt
import pandas as pd

df_iAF987 = pd.read_csv('../../../DeepEnzyme/Data/Fig/BiGG/iAF987.csv')

data_iAF987 = []
for j in df_iAF987.index:
    data_iAF987.append(df_iAF987['Kcat_lgo10'][j])

plt.rcParams.update({'font.size': 12})
plt.hist(data_iAF987, bins=75, color='#5875A7', edgecolor='black')

plt.xlabel('Predicted $log$$_\mathregular{10}$($k$$_\mathregular{cat}$)')
plt.ylabel('Frequency')

plt.show()
#plt.savefig("../../../DeepEnzyme/Results/Figures/Fig6b.pdf", dpi=600, bbox_inches='tight')
