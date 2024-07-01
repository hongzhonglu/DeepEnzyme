import json
import math
import numpy as np
import pickle
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch import nn, optim
import timeit
from Code.Model.DeepEnzyme import DeepEnzyme

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def load_proteinadjacencies(file_name, dtype):
    return [dtype(d.toarray()).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)  # np.random.seed()函数用于生成指定随机数
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


def cat(list1, list2):
    for i in list2:
        list1.append(i)
    return list1


class Trainer(object):
    def __init__(self, model, lr, weight_decay):
        self.model = model
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.9)
        self.loss = nn.MSELoss().to(device)

    def train(self, dataset, layer_output, dropout):
        # np.random.shuffle(dataset)
        self.model.train(True)
        train_loss_epoch_total = 0
        SAE = 0
        N = len(dataset)
        trainY, trainPredict = [], []
        for data in dataset:
            self.optimizer.zero_grad()
            output = self.model(data, layer_output, dropout)
            loss = self.loss(output, data[-1])
            loss.backward()
            self.optimizer.step()
            train_loss_epoch_total += loss.item()

            #correct_kcat = math.log10(math.pow(2, data[-1].to('cpu').data.numpy()))
            #predict_kcat = math.log10(math.pow(2, output.to('cpu').data.numpy()))
            correct_kcat = float(data[-1].to('cpu').data.numpy())
            predict_kcat = float(output.to('cpu').data.numpy())
            trainY.append(correct_kcat)
            trainPredict.append(predict_kcat)

            SAE += np.abs(predict_kcat - correct_kcat)

        MAE = SAE / N
        rmse = np.sqrt(mean_squared_error(trainY, trainPredict))
        R2 = r2_score(trainY, trainPredict)
        train_loss = train_loss_epoch_total / len(dataset)
        return train_loss, MAE, rmse, R2


class Validater(object):
    def __init__(self, model):
        self.model = model
        self.loss = nn.MSELoss().to(device)

    def val(self, dataset, layer_output, dropout):
        # self.model.eval()
        self.model.train(False)
        val_loss_epoch_total = 0
        N = len(dataset)
        SAE = 0
        testY, testPredict = [], []
        for data in dataset:
            output = self.model(data, layer_output, dropout)
            loss = self.loss(output, data[-1])
            val_loss_epoch_total += loss.item()

            #correct_kcat = math.log10(math.pow(2, data[-1].to('cpu').data.numpy()))
            #predict_kcat = math.log10(math.pow(2, output.to('cpu').data.numpy()))
            correct_kcat = float(data[-1].to('cpu').data.numpy())
            predict_kcat = float(output.to('cpu').data.numpy())
            testY.append(correct_kcat)
            testPredict.append(predict_kcat)

            SAE += np.abs(predict_kcat - correct_kcat)

        MAE = SAE / N
        rmse = np.sqrt(mean_squared_error(testY, testPredict))
        R2 = r2_score(testY, testPredict)
        val_loss = val_loss_epoch_total / len(dataset)
        return val_loss, MAE, rmse, R2


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset, layer_output, dropout):
        # self.model.eval()
        self.model.train(False)
        N = len(dataset)
        SAE = 0
        testY, testPredict = [], []
        for data in dataset:
            output = self.model(data, layer_output, dropout)

            #correct_kcat = math.log10(math.pow(2, data[-1].to('cpu').data.numpy()))
            #predict_kcat = math.log10(math.pow(2, output.to('cpu').data.numpy()))
            correct_kcat = float(data[-1].to('cpu').data.numpy())
            predict_kcat = float(output.to('cpu').data.numpy())
            testY.append(correct_kcat)
            testPredict.append(predict_kcat)

            SAE += np.abs(predict_kcat - correct_kcat)

        MAE = SAE / N
        rmse = np.sqrt(mean_squared_error(testY, testPredict))
        R2 = r2_score(testY, testPredict)
        return MAE, rmse, R2

    def save_MAEs(self, MAEs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, MAEs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


def main():
    lr = 0.001
    iteration = 200
    weight_decay = 1e-6
    dropout = 0.3
    dim = 64
    layer_output = 3
    hidden_dim1 = 64
    hidden_dim2 = 64
    nhead = 4
    hid_size = 64
    layers_trans = 3

    dir_input = '../../../DeepEnzyme/Data/Input/'
    #proteinadjacencies_0612/smileadjacencies_0612/sequences_0612 can be downloaded from https://figshare.com/articles/dataset/DeepEnzyme/25771062
    fingerprint = load_tensor(dir_input + 'fingerprint_0612', torch.LongTensor)
    smileadjacencies = load_tensor(dir_input + 'smileadjacencies_0612', torch.FloatTensor)
    sequences = load_tensor(dir_input + 'sequences_0612', torch.LongTensor)
    proteinadjacencies = np.load(dir_input + 'proteinadjacencies_0612.npy', allow_pickle=True)
    kcat_tensor = load_tensor(dir_input + 'logkcat_0612', torch.FloatTensor)
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict_0612.pickle')
    word_dict = load_pickle(dir_input + 'sequence_dict_0612.pickle')
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    dataset = list(zip(fingerprint, smileadjacencies, sequences, proteinadjacencies, kcat_tensor))
    dataset = shuffle_dataset(dataset, 666)
    #np.random.shuffle(dataset)
    dataset_train, dataset_ = split_dataset(dataset, 0.8)
    dataset_val, dataset_test = split_dataset(dataset_, 0.5)

    model = DeepEnzyme(n_fingerprint, dim, n_word, layer_output, hidden_dim1, hidden_dim2, dropout, nhead, hid_size,
                        layers_trans).to(device)
    trainer = Trainer(model, lr, weight_decay)
    validater = Validater(model)
    tester = Tester(model)

    file_MAEs = '../../../DeepEnzyme/Result/Output/dim64_lr001_E200_head4_drop3_seed666.txt'
    file_model = '../../../DeepEnzyme/Result/Output/dim64_lr001_E200_head4_drop3_seed666'
    MAEs = 'Epoch\tTime(sec)\tLoss_train\tMAE_train\tRMSE_train\tR2_train\tloss_val\tMAE_val\tMAE_test' \
        '\tRMSE_val\tRMSE_test\tR2_val\tR2_test\tLr'
    with open(file_MAEs, 'w') as f:
        f.write(MAEs + '\n')

    print('Training...')
    print(MAEs)
    start = timeit.default_timer()

    for epoch in range(1, iteration + 1):

        loss_train, MAE_train, RMSE_train, R2_train = trainer.train(dataset_train, layer_output, dropout)
        loss_val, MAE_val, RMSE_val, R2_val = validater.val(dataset_val, layer_output, dropout)
        MAE_test, RMSE_test, R2_test = tester.test(dataset_test, layer_output, dropout)

        if epoch // 10 > 0 and epoch % 10 > 0:
            trainer.scheduler.step()

        end = timeit.default_timer()
        time = end - start

        MAEs = [epoch, round(time, 4), round(loss_train, 4), round(MAE_train, 4), round(RMSE_train, 4),
                round(R2_train, 4), round(loss_val, 4), round(MAE_val, 4), round(MAE_test, 4), round(RMSE_val, 4),
                round(RMSE_test, 4), round(R2_val, 4), round(R2_test, 4), trainer.optimizer.state_dict()['param_groups'][0]['lr']]

        tester.save_MAEs(MAEs, file_MAEs)
        tester.save_model(model, file_model)

        print('\t'.join(map(str, MAEs)))


if __name__ == "__main__":
    main()
