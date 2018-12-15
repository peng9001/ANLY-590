import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torch.autograd import Variable
import string
import torch.nn as nn
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
import datetime
all_characters = string.printable
n_characters = len(all_characters)

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

def generate(decoder, prime_str='A', predict_len=100, temperature=0.8, cuda=False):
    hidden = decoder.init_hidden(1)
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:, p], hidden)

    inp = prime_input[:, -1]

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

    return predicted

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model="gru", n_layers=1):
        super(CharRNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.model == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def forward2(self, input, hidden):
        encoded = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        if self.model == "lstm":
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

# def calculate_word_acc(result_str):
#     tmp_str = word_tokenize(result_str)[:-1]
#     spell = SpellChecker()
#     misspelled = spell.unknown(tmp_str)
# #    print(misspelled)
#     error_rate = len(misspelled)/len(tmp_str)
#     return 1 - error_rate

def calculate_word_acc(result_str):
    # word-level
    tmp_str = word_tokenize(result_str)[:-1]
    spell = SpellChecker()
    misspelled = spell.unknown(tmp_str)
    # print(misspelled)
    error_rate = len(misspelled)/len(tmp_str)
    error_length = sum([len(each) for each in misspelled])
    total_length = sum([len(each) for each in tmp_str])
    error_rate_char = error_length/total_length
    return 1 - error_rate, 1 - error_rate_char


num_hidden = [5,20,50,100,200]

for i,num in enumerate(num_hidden):
    locals()['layer_'+str(i+1)] = pd.read_csv('Model_Loss_2LayersGRU_'+str(num)+'_hidden_1000_epoch.csv').iloc[:,[1,2,3]]
    locals()['model_'+str(i+1)] = 'emi_lyrics_2LayersGRU_'+str(num)+'_hidden_1000_epoch.pt'
##########画loss图###########

# plt.figure()
# plt.title('Result Analysis in terms of number of hidden size')
# plt.plot(layer_1.epoch, layer_1.loss, color='green', label='5 hidden size',linestyle='-')
# plt.plot(layer_2.epoch, layer_2.loss, color='red', label='20 hidden size')
# plt.plot(layer_3.epoch, layer_3.loss, color='blue', label='50 hidden size')
# plt.plot(layer_4.epoch, layer_4.loss, color='skyblue', label='100 hidden size')
# plt.plot(layer_5.epoch, layer_5.loss, color='yellow', label='200 hidden size')
# plt.legend() # 显示图例
# plt.xlabel('number of epochs')
# plt.ylabel('training loss')
# plt.show()



# Run as standalone script

# prime_str = 'I'
# predict_len = 1000
# temperature = 0.3
# cuda = False

# 注释代码
# word_acc = []
# char_acc = []
# Time = []
#
# for i in range(1,4):
#     model = eval('model'+str(i))
#     decoder = torch.load(model)
#
#     Lyrics = ''
#     for _ in range(50):
#         s = generate(decoder, prime_str, predict_len, temperature)+'\n'
#         Lyrics += s
#
#     a, b = calculate_word_acc(Lyrics)
#     word_acc.append(a)
#     char_acc.append(b)
#     df = eval('layer_'+str(i))
#     timee = str(datetime.timedelta(seconds=int(layer_1.loc[0,'time'])))
#     Time.append(timee)
#
# print(word_acc,char_acc,Time)

# 导入Lyrics并生成accuracy的table
Lyrics_table = pd.read_csv('./table_acc_hidden.csv',header=0)
cols=['Lyrics_5hidden','Lyrics_20hidden','Lyrics_50hidden','Lyrics_100hidden','Lyrics_200hidden']
Lyrics_table=Lyrics_table.ix[:,cols]
print(Lyrics_table.head(3))
print(Lyrics_table.columns.values)
word_acc = []
char_acc = []
Time = []
for i in range(1,6):
    Lyrics = ''
    for j in range(Lyrics_table.shape[0]):
        Lyrics += Lyrics_table.iloc[j, i-1] +' '
    a, b = calculate_word_acc(Lyrics)
    word_acc.append(a)
    char_acc.append(b)
    df = eval('layer_'+str(i))
    timee = str(datetime.timedelta(seconds=int(df.loc[0,'time'])))
    Time.append(timee)

table = pd.DataFrame({'word_acc':word_acc,'char_acc':char_acc,'Time':Time},index=Lyrics_table.columns.values)
table.to_csv('table_model_hidden.csv')

