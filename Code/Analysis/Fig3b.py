import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


#with open('../DeepEnzyme/Results/Output_results/training_onlystructure.txt') as training_onlystructure:
#    data = training_onlystructure.readlines()
#    training_onlystructure.close()
#training_onlystructure = data[-1].split('\t')[-4]


with open('../../../DeepEnzyme/Results/Output_results/dim64_lr001_E100_head4_drop3_0612_seed666.txt') as input:
    data = input.readlines()
    input.close()
all_data = data[-1].split('\t')[-4]

df = pd.read_csv('../../../DeepEnzyme/Data/Fig/DLTKcat_out.csv')
ex = []
for i in df['log_kcat']:
    ex.append(i)
pre = []
for i in df['pred_log10kcat']:
    pre.append(i)

DLTKcat = np.sqrt(mean_squared_error(ex, pre))

# data from paper
DLKcat = 1.1254
TurNuP = 0.9274

x = ['DeepEnzyme', 'DLTKcat', 'DLKcat', 'TurNuP']
y = [float(all_data), DLTKcat, DLKcat, TurNuP]

colors = ['#2F4996', '#2F4996', '#2F4996', '#2F4996']

plt.bar(x, y, color=colors, width=0.3)
plt.title('test set RMSE')
plt.ylabel('RMSE')

plt.show()
#plt.savefig("../../../DeepEnzyme/Results/Figures/Fig3b.pdf", dpi=600, bbox_inches='tight')
