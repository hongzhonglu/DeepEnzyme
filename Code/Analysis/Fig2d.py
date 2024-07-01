import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

df = pd.read_csv('../../../DeepEnzyme/Data/Fig/Fig2.csv')

EC_1_pre, EC_1_exp = [], []
EC_2_pre, EC_2_exp = [], []
EC_3_pre, EC_3_exp = [], []
EC_4_pre, EC_4_exp = [], []
EC_5_pre, EC_5_exp = [], []
EC_6_pre, EC_6_exp = [], []

for i in df.index:
    if df['EC_number'][i].split('.')[0] == '1':
        EC_1_pre.append(df['Predicted logkcat'][i])
        EC_1_exp.append(df['experiment logkcat'][i])
    if df['EC_number'][i].split('.')[0] == '2':
        EC_2_pre.append(df['Predicted logkcat'][i])
        EC_2_exp.append(df['experiment logkcat'][i])
    if df['EC_number'][i].split('.')[0] == '3':
        EC_3_pre.append(df['Predicted logkcat'][i])
        EC_3_exp.append(df['experiment logkcat'][i])
    if df['EC_number'][i].split('.')[0] == '4':
        EC_4_pre.append(df['Predicted logkcat'][i])
        EC_4_exp.append(df['experiment logkcat'][i])
    if df['EC_number'][i].split('.')[0] == '5':
        EC_5_pre.append(df['Predicted logkcat'][i])
        EC_5_exp.append(df['experiment logkcat'][i])
    if df['EC_number'][i].split('.')[0] == '6':
        EC_6_pre.append(df['Predicted logkcat'][i])
        EC_6_exp.append(df['experiment logkcat'][i])

pearson_corr_1, p_value_1 = pearsonr(EC_1_pre, EC_1_exp)
R2_1 = r2_score(EC_1_exp, EC_1_pre)

pearson_corr_2, p_value_2 = pearsonr(EC_2_pre, EC_2_exp)
R2_2 = r2_score(EC_2_exp, EC_2_pre)

pearson_corr_3, p_value_3 = pearsonr(EC_3_pre, EC_3_exp)
R2_3 = r2_score(EC_3_exp, EC_3_pre)

pearson_corr_4, p_value_4 = pearsonr(EC_4_pre, EC_4_exp)
R2_4 = r2_score(EC_4_exp, EC_4_pre)

pearson_corr_5, p_value_5 = pearsonr(EC_5_pre, EC_5_exp)
R2_5 = r2_score(EC_5_exp, EC_5_pre)

pearson_corr_6, p_value_6 = pearsonr(EC_6_pre, EC_6_exp)
R2_6 = r2_score(EC_6_exp, EC_6_pre)

x = ['EC 1.', 'EC 2.', 'EC 3.', 'EC 4.', 'EC 5.', 'EC 6.']
y = [R2_1, R2_2, R2_3, R2_4, R2_5, R2_6]

plt.bar(x, y, color='#5875A7', width=0.5)

plt.title('${R^2}$ of different EC number')
plt.ylabel('${R^2}$')

plt.show()
# plt.savefig("../../../DeepEnzyme/Results/Figures/Fig2d.pdf", dpi=600, bbox_inches='tight')
