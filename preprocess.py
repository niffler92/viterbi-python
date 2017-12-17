import os

import pandas as pd
import numpy as np


def get_unigram_dict(filepath):
    unigram_dict = {}

    with open(filepath, 'r') as unigram:
        lines = unigram.read()
        lines = lines.split()
        for idx, line in enumerate(lines):
            if idx % 2 == 0:
                unigram_dict[line] = float(lines[idx+1])

    return unigram_dict


def get_bigram_dict(filepath):
    bigram_dict = {}

    with open(filepath, 'r') as bigram_txt:
        for line in bigram_txt:
            line = line.split()
            bigram_dict[line[0]] = {}

    with open(filepath, 'r') as bigram_txt:
        for line in bigram_txt:
            line = line.split()
            bigram_dict[line[0]][line[1]] = float(line[2])

    return bigram_dict


def get_phoneme_dict(filepath):
    phoneme_dict = {}

    with open(filepath, 'r') as dict_txt:
        for line in dict_txt:
            line = line.split()
            phonemes = []
            word = line[0]
            for i in range(1, len(line)):
                phonemes.append(line[i])

            if(word in phoneme_dict.keys()):
                phoneme_dict[word] = [phoneme_dict[word], phonemes]
            else:
                phoneme_dict[word] = phonemes

    return phoneme_dict


def get_hmm_dict(filepath):
    if os.path.exists("./data/hmm_data.npy"):
        hmm_data = np.load("./data/hmm_data.npy").item()
        hmm_dict = hmm_data["hmm_dict"]
    else:
        hmm_dict = {}
        hmm_txt = open(filepath, 'r')
        num_state = 3 # Each phoneme has 3 states except 'sp' (idx==20)
        for idx in range(21):  # 21: Number of unique phonemes
            pronun_word = hmm_txt.readline().split('"')[1]
            hmm_dict[pronun_word] = {}

            hmm_txt.readline()

            Numstates = hmm_txt.readline().split()
            hmm_dict[pronun_word][Numstates[0]] = Numstates[1]

            if idx==20:
                num_state = 1

            for st_idx in range(num_state):
                #state number
                state = hmm_txt.readline().split()[1]
                hmm_dict[pronun_word][state] = {}


                Num_Mixes = hmm_txt.readline().split()
                hmm_dict[pronun_word][state][Num_Mixes[0]] = Num_Mixes[1]
                hmm_dict[pronun_word][state]['<MIXTURES>'] = {}

                for mix_idx in range(1,11):

                    mixture = hmm_txt.readline().split()

                    hmm_dict[pronun_word][state]['<MIXTURES>'][mixture[1]] = {}
                    hmm_dict[pronun_word][state]['<MIXTURES>'][mixture[1]][mixture[0]] = mixture[2]

                    #Mean
                    mean_dim = hmm_txt.readline().split()
                    mean_num = hmm_txt.readline().split()
                    hmm_dict[pronun_word][state]['<MIXTURES>'][mixture[1]][mean_dim[0]] = mean_num

                    #Variance
                    variance_dim = hmm_txt.readline().split() #input dimension = 39
                    variance_num = hmm_txt.readline().split()
                    hmm_dict[pronun_word][state]['<MIXTURES>'][mixture[1]][variance_dim[0]] = variance_num

                    #GConst
                    g_const = hmm_txt.readline().split()
                    hmm_dict[pronun_word][state]['<MIXTURES>'][mixture[1]][g_const[0]] = g_const[1]

            #Transposition Probability
            hmm_txt.readline()
            trans_prob=[]

            # a matrix
            if idx != 20:
                for trans_idx in range(1,6):
                    trans_prob.append(hmm_txt.readline().split())
            else:
                for trans_idx in range(1,4):
                    trans_prob.append(hmm_txt.readline().split())

            hmm_dict[pronun_word]['<TRANSP>'] = trans_prob
            #ENDHMM
            hmm_txt.readline()
        hmm_txt.close()

    return hmm_dict


def get_test_data(test_dir):
    test_data = {}

    for folder in os.listdir(test_dir):
        for folder2 in os.listdir("{}/{}".format(test_dir, folder)):
            for dirpath, _, files in os.walk("{}/{}/{}".format(test_dir, folder, folder2)):
                for file in files:
                    with open(os.path.join(dirpath, file)) as txt:
                        label = file.split(".")[0]
                        shape = txt.readline().split()
                        test_data[label] = {'length': int(shape[0]), 'n_features': int(shape[1])}

                        mfcc = np.ndarray((int(shape[0]), int(shape[1])))
                        for idx, line in enumerate(txt.readlines()):
                            mfcc[idx] = np.array(line.split()).astype(float)

                        test_data[label]['mfcc'] = mfcc

    return test_data


# For test
if __name__ == '__main__':
    unigram_dict = get_unigram_dict("./data/unigram.txt")
    bigram_dict = get_bigram_dict("./data/bigram.txt")
    phoneme_dict = get_phoneme_dict("./data/dictionary.txt")
    hmm_dict = get_hmm_dict("./data/hmm.txt")
    test_data = get_test_data("./data/tst")

    print(len(unigram_dict), len(bigram_dict), len(phoneme_dict), len(hmm_dict), len(test_data))
