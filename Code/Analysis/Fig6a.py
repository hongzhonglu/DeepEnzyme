import matplotlib.pyplot as plt
import pandas as pd
from glob import glob

csv_files = glob('../../../DeepEnzyme/Data/Fig/BiGG/*')
data = []
for i in csv_files:
    if 'seq' not in i and 'SMILES' not in i and 'csv' in i:
        df = pd.read_csv(i)
        for j in df.index:
            data.append(df['Kcat_lgo10'][j])

plt.rcParams.update({'font.size': 12})
plt.hist(data, bins=65, color='#5875A7', edgecolor='black')

plt.xlabel('Predicted $log$$_\mathregular{10}$($k$$_\mathregular{cat}$)')
plt.ylabel('Frequency')

plt.show()
#plt.savefig("../../../DeepEnzyme/Results/Figures/Fig6a.pdf", dpi=600, bbox_inches='tight')
