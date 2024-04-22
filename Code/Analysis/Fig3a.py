import matplotlib.pyplot as plt

with open('../DeepEnzyme/Results/Output_results/training_onlystructure.txt') as training_onlystructure:
    data = training_onlystructure.readlines()
    training_onlystructure.close()
training_onlystructure = data[-1].split('\t')[-2]

with open('../DeepEnzyme/Results/Output_results/dim64_lr001_E100_head4_drop3_0612_seed666.txt') as input:
    data = input.readlines()
    input.close()
all_data = data[-1].split('\t')[-2]

# data from paper
DLKcat = 0.4386
TurNuP = 0.40

x = ['DeepEnzyme', 'only-Structure', 'DLKcat', 'TurNuP']
y = [float(all_data), float(training_onlystructure), DLKcat, DLKcat]

colors = ['#2F4996', '#2F4996', '#2F4996', '#2F4996']

plt.bar(x, y, color=colors, width=0.3)
plt.title('test set ${R^2}$')
plt.ylabel('${R^2}$')

plt.show()
#plt.savefig("../../figure/Fig3a.pdf", dpi=600, bbox_inches='tight')
