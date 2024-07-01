import matplotlib.pyplot as plt

with open('../../../DeepEnzyme/Results/Output_results/training_onlysequences.txt') as training_onlysequences:
    data = training_onlysequences.readlines()
    training_onlysequences.close()
training_onlysequences = data[-1].split('\t')[-2]

with open('../../../DeepEnzyme/Results/Output_results/training_onlystructure.txt') as training_onlystructure:
    data = training_onlystructure.readlines()
    training_onlystructure.close()
training_onlystructure = data[-1].split('\t')[-2]

with open('../../../DeepEnzyme/Results/Output_results/dim64_lr001_E100_head4_drop3_0612_seed666.txt') as input:
    data = input.readlines()
    input.close()
all_data = data[-1].split('\t')[-2]

x = ['DeepEnzyme', 'Only-Structure', 'Only-Sequence']
y = [float(all_data), float(training_onlystructure), float(training_onlysequences)]

colors = ['#5875A7', '#5875A7', '#5875A7']

plt.bar(x, y, color=colors, width=0.25)

plt.title('test set ${R^2}$')
plt.ylabel('${R^2}$')

plt.show()
# plt.savefig("../../../DeepEnzyme/Results/Figures/Fig2e.pdf", dpi=600, bbox_inches='tight')
