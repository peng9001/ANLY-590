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



layer_1 = pd.read_csv('Model_Loss_1LayersGRU_100_hidden_1000_epoch.csv').iloc[:,[1,2,3]]
layer_2 = pd.read_csv('Model_Loss_2LayersGRU_100_hidden_1000_epoch.csv').iloc[:,[1,2,3]]
layer_3 = pd.read_csv('Model_Loss_3LayersGRU_100_hidden_1000_epoch.csv').iloc[:,[1,2,3]]

##########画loss图###########

# plt.figure()
# plt.title('Result Analysis')
# plt.plot(layer_1.epoch, layer_1.loss, color='green', label='1 layer GRU',linestyle='-')
# plt.plot(layer_2.epoch, layer_2.loss, color='red', label='2 layer GRU')
# plt.plot(layer_3.epoch, layer_3.loss, color='blue', label='3 layer GRU')
# plt.legend() # 显示图例
#
# plt.xlabel('number of epochs')
# plt.ylabel('training loss')
# plt.show()



# Run as standalone script
model1 = 'emi_lyrics_1LayersGRU_100_hidden_1000_epoch.pt'
model2 = 'emi_lyrics_2LayersGRU_100_hidden_1000_epoch.pt'
model3 = 'emi_lyrics_3LayersGRU_100_hidden_1000_epoch.pt'
prime_str = 'I'
predict_len = 1000
temperature = 0.3
cuda = False

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
# Lyrics_table = pd.read_csv('../table_acc.csv')

# word_acc = []
# char_acc = []
# Time = []
# for i in range(1,4):
#     Lyrics = ''
#     for j in range(Lyrics_table.shape[0]):
#         Lyrics += Lyrics_table.iloc[j, i] +' '
#     a, b = calculate_word_acc(Lyrics)
#     word_acc.append(a)
#     char_acc.append(b)
#     df = eval('layer_'+str(i))
#     timee = str(datetime.timedelta(seconds=int(df.loc[0,'time'])))
#     Time.append(timee)
#
# table = pd.DataFrame({'word_acc':word_acc,'char_acc':char_acc,'Time':Time})
# table.to_csv('table_model.csv')

str200 = """
Hey
but i just don't give a fuck
i wanna take me too, she has told you shot
'cause you give'em the new rist so we can looks like a fuckin' colla
don't want you no fame, want you no gut
so they say they're just a chuck
and when i go to you, but it's just the type of my back
slam with me, let on a soldier who wanna rock the heck
when i just choke of it, it subla
cause i smiles and shook this head while we going to rap
when i'm slim shady
so you just get one control
come on a second shot 
we came off the flow
"""
str100 = """
Hey
you money to beful
i shit i don't got a little back, i hear the stupid
so mad you should tell it and i do it
imma be really faking in the honesher doubf
custom what you know you wanna man through the peach
but i can screamings around it, so they ha would it gonna see me
and i got down, down it the past ruthing they talk it that's every gone
what you let's beginning a hole it, i feel hours againma some days and vister
and i up and let pocket your channipositions that with millions
so i can 'neal
"""
print('100',calculate_word_acc(str100))
print('200',calculate_word_acc(str200))

