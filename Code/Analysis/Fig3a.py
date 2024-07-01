import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

#with open('../DeepEnzyme/Results/Output_results/training_onlystructure.txt') as training_onlystructure:
#    data = training_onlystructure.readlines()
#    training_onlystructure.close()
#training_onlystructure = data[-1].split('\t')[-2]

with open('../../../DeepEnzyme/Results/Output_results/dim64_lr001_E100_head4_drop3_0612_seed666.txt') as input:
    data = input.readlines()
    input.close()
all_data = data[-1].split('\t')[-2]

df = pd.read_csv('../../../DeepEnzyme/Data/Fig/DLTKcat_out.csv')
ex = []
for i in df['log_kcat']:
    ex.append(i)
pre = []
for i in df['pred_log10kcat']:
    pre.append(i)
DLTKcat = r2_score(ex, pre)

# data from paper
DLKcat = 0.4386
TurNuP = 0.40

x = ['DeepEnzyme', 'DLTKcat', 'DLKcat', 'TurNuP']
y = [float(all_data), DLTKcat, DLKcat, TurNuP]

colors = ['#2F4996', '#2F4996', '#2F4996', '#2F4996']

plt.bar(x, y, color=colors, width=0.3)
plt.title('test set ${R^2}$')
plt.ylabel('${R^2}$')

plt.show()
#plt.savefig("../../../DeepEnzyme/Results/Figures/Fig3a.pdf", dpi=600, bbox_inches='tight')
