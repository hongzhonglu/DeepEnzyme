import matplotlib.pyplot as plt

with open('../../../DeepEnzyme/Results/Output_results/random_1.txt') as input_1:
    data = input_1.readlines()
    input_1.close()
input_1_test = data[-1].split('\t')[-2]
input_1_train = data[-1].split('\t')[5]

with open('../../../DeepEnzyme/Results/Output_results/random_2.txt') as input_2:
    data = input_2.readlines()
    input_2.close()
input_2_test = data[-1].split('\t')[-2]
input_2_train = data[-1].split('\t')[5]

with open('../../../DeepEnzyme/Results/Output_results/random_3.txt') as input_3:
    data = input_3.readlines()
    input_3.close()
input_3_test = data[-1].split('\t')[-2]
input_3_train = data[-1].split('\t')[5]

with open('../../../DeepEnzyme/Results/Output_results/random_4.txt') as input_4:
    data = input_4.readlines()
    input_4.close()
input_4_test = data[-1].split('\t')[-2]
input_4_train = data[-1].split('\t')[5]

with open('../../../DeepEnzyme/Results/Output_results/random_5.txt') as input_5:
    data = input_5.readlines()
    input_5.close()
input_5_test = data[-1].split('\t')[-2]
input_5_train = data[-1].split('\t')[5]

test = [float(input_1_test), float(input_2_test), float(input_3_test), float(input_4_test), float(input_5_test)]
train = [float(input_1_train), float(input_2_train), float(input_3_train), float(input_4_train), float(input_5_train)]

plt.style.use('seaborn-ticks')

fig, ax = plt.subplots(facecolor='white')

boxplot = plt.boxplot([test, train], labels=['Testing  ${R^2}$', 'Training  ${R^2}$'], patch_artist=True)

colors = ['#2C4895', '#2C4895']
for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')

plt.ylabel('${R^2}$', fontsize=14)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()
#plt.savefig("../../../DeepEnzyme/Results/Figures/Fig2f.pdf", dpi=600, bbox_inches='tight')
