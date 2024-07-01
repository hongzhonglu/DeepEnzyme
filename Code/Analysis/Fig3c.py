import numpy as np
import pandas as pd
from os.path import join
from sklearn.metrics import r2_score
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

df = pd.read_csv('../../../DeepEnzyme/Data/Fig/Fig2.csv')
df_DLTKcat = pd.read_csv('../../../DeepEnzyme/Data/Fig/DLTKcat_out.csv')
data_test = pd.read_pickle(join("../../../DeepEnzyme/Data/Fig/test_df_kcat.pkl"))
pred_y = np.load(
    join("../../../DeepEnzyme/Data/Fig/y_test_pred_xgboost_ESM1b_ts_DRFP_mean.npy"))
test_y = np.load(
    join("../../../DeepEnzyme/Data/Fig/y_test_true_xgboost_ESM1b_ts_DRFP_mean.npy"))

data_test["y_true"] = test_y
data_test["y_pred"] = pred_y
data_test["max_ident"] = np.nan

for ind in data_test.index:
    try:
        with open(join("../../../DeepEnzyme/Data/Fig/kcat_ident/test_seq" + str(ind) + ".txt")) as f:
            ident = f.readlines()
            ident = float(ident[0])

        data_test["max_ident"][ind] = ident
    except FileNotFoundError:
        pass

data_test_DLkcat = pd.read_pickle(join("../../../DeepEnzyme/Data/Fig/df_pred.pkl"))

splits = ["0-50%", "50-90%", '90-100%']
lower_bounds = [0, 50, 90]
upper_bounds = [50, 90, 100]

points1, points1_sim = [], []
points2, points2_sim = [], []
n_points1, n_points2 = [], []
n_points1_sim, n_points2_sim = [], []
y_TurNuP = []
y_DLkcat = []

for i, split in enumerate(splits):
    lb, ub = lower_bounds[i], upper_bounds[i]
    help_df = data_test.loc[data_test["max_ident"] >= lb].loc[data_test["max_ident"] <= ub]
    y_true = np.array(help_df["y_true"])
    y_pred = np.array(help_df["y_pred"])
    n_kcat = len(y_pred)
    R2_TurNuP = r2_score(y_true, y_pred)
    help_df = data_test_DLkcat.loc[data_test_DLkcat["max_ident"] >= lb].loc[data_test_DLkcat["max_ident"] <= ub]
    y_true = np.array(help_df["y_true"])
    y_pred = np.array(help_df["y_pred"])
    n_DLkcat = len(y_pred)
    R2_DLkcat = r2_score(y_true, y_pred)
    #print("%s TurNuP: R2:%s" % (split, R2_TurNuP))
    #print("%s DLKcat: R2:%s" % (split, R2_DLkcat))
    y_TurNuP.append(R2_TurNuP)
    y_DLkcat.append(R2_DLkcat)

DeepEnzyme_0_50_exp = []
DeepEnzyme_0_50_pre = []
DeepEnzyme_50_90_exp = []
DeepEnzyme_50_90_pre = []
DeepEnzyme_90_100_exp = []
DeepEnzyme_90_100_pre = []

for i in df.index:
    if 0 <= df['score'][i] <= 0.5:
        DeepEnzyme_0_50_exp.append(df['experiment logkcat'][i])
        DeepEnzyme_0_50_pre.append(df['Predicted logkcat'][i])
    if 0.5 <= df['score'][i] <= 0.9:
        DeepEnzyme_50_90_exp.append(df['experiment logkcat'][i])
        DeepEnzyme_50_90_pre.append(df['Predicted logkcat'][i])
    if 0.9 <= df['score'][i] <= 1:
        DeepEnzyme_90_100_exp.append(df['experiment logkcat'][i])
        DeepEnzyme_90_100_pre.append(df['Predicted logkcat'][i])

DeepEnzyme_0_50 = r2_score(DeepEnzyme_0_50_exp, DeepEnzyme_0_50_pre)
DeepEnzyme_50_90 = r2_score(DeepEnzyme_50_90_exp, DeepEnzyme_50_90_pre)
DeepEnzyme_90_100 = r2_score(DeepEnzyme_90_100_exp, DeepEnzyme_90_100_pre)

ex_DLTKcat_0_5 = []
ex_DLTKcat_5_9 = []
ex_DLTKcat_9_10 = []
pre_DLTKcat_0_5 = []
pre_DLTKcat_5_9 = []
pre_DLTKcat_9_10 = []
for i in df_DLTKcat['EC'].index:
    if df_DLTKcat['aln'][i] <= 0.5:
        ex_DLTKcat_0_5.append(df_DLTKcat['log_kcat'][i])
        pre_DLTKcat_0_5.append(df_DLTKcat['pred_log10kcat'][i])
    if 0.5 <= df_DLTKcat['aln'][i] <= 0.9:
        ex_DLTKcat_5_9.append(df_DLTKcat['log_kcat'][i])
        pre_DLTKcat_5_9.append(df_DLTKcat['pred_log10kcat'][i])
    if 0.9 <= df_DLTKcat['aln'][i] <= 1:
        ex_DLTKcat_9_10.append(df_DLTKcat['log_kcat'][i])
        pre_DLTKcat_9_10.append(df_DLTKcat['pred_log10kcat'][i])

r_DLTKcat_0_5 = r2_score(ex_DLTKcat_0_5, pre_DLTKcat_0_5)
r_DLTKcat_5_9 = r2_score(ex_DLTKcat_5_9, pre_DLTKcat_5_9)
r_DLTKcat_9_10 = r2_score(ex_DLTKcat_9_10, pre_DLTKcat_9_10)

x = ['0-50%', '50%-90%', '90%-100%']
x = np.arange(len(x))
width = 0.1

y1 = [DeepEnzyme_0_50, DeepEnzyme_50_90, DeepEnzyme_90_100]
y2 = y_TurNuP
y3 = [r_DLTKcat_0_5, r_DLTKcat_5_9, r_DLTKcat_9_10]
y4 = y_DLkcat

fig, ax = plt.subplots()

rects1 = ax.bar(x - width, y1, width, color='#5875A7', label='DeepEnzyme')
rects2 = ax.bar(x, y2, width, color='#CD885D', label='TurNuP')
rects3 = ax.bar(x + width, y3, width, color='#5F9E98', label='DLTKcat')
rects4 = ax.bar(x + width*2, y4, width, color='#7894BC', label='DLKcat')

ax.set_xticks(x)
ax.set_xticklabels(['0-50%', '50%-90%', '90%-100%'])
ax.set_xlabel('Enzyme sequence identity')
ax.set_ylabel('${R^2}$')

ax.legend(loc='lower right')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')

plt.tight_layout()

plt.show()
#plt.savefig("../../../DeepEnzyme/Results/Figures/Fig3c.pdf", dpi=600, bbox_inches='tight')
