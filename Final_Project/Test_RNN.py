import torch
import os
from torch.autograd import Variable
import string
import torch.nn as nn
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

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

# Run as standalone script
if __name__ == '__main__':
    filename1 = './model/emi_lyrics_4Layers_LSTM.pt'
    filename2 = './model/emi_lyrics_4Layers_GRU.pt'
    filename3 = './model/emi_lyrics_2Layers_GRU.pt'
    filename4 = './model/emi_lyrics_2Layers_GRU_1000e_128b.pt'
    prime_str = 'I'
    predict_len = 1000
    temperature = 0.3
    cuda = False
    # for filename in [filename1, filename2, filename3, filename4]:
    #
    #     decoder = torch.load(filename)
    #     #出结果
    #     print(filename, '\n', generate(decoder, prime_str, predict_len, temperature),'\n\n')

        #check the accuracy of predicted words
        # Lyrics = ''
        # for _ in range(50):
        #     s = generate(decoder, prime_str, predict_len, temperature)+'\n'
        #     Lyrics += s

        # print(filename, calculate_word_acc(Lyrics))

    decoder = torch.load(filename2)
    for _ in range(1):
        print(generate(decoder, prime_str, predict_len, temperature), '\n\n')

