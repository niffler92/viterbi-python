import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from preprocess import *
from inference import *
from hmm import HMM


def label_to_word(label):
    word_dict = {'1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
                '8': 'eight', '9': 'nine', 'o': 'oh', 'z': 'zero'}

    return [word_dict[label] for label in label]


def main():
    test_data = get_test_data("./data/tst")
    unigram_dict = get_unigram_dict("./data/unigram.txt")
    bigram_dict = get_bigram_dict("./data/bigram.txt")
    phoneme_dict = get_phoneme_dict("./data/dictionary.txt")
    hmm_dict = get_hmm_dict("./data/hmm.txt")

    hmm = HMM(hmm_dict)

    y_preds = []
    y_trues = list(test_data.keys())


    #Loop
    #for label in y_trues:
    #    words_pred =  continuous_recognition(unigram_dict, bigram_dict, phoneme_dict, hmm, test_data[label]['mfcc'])
    #    y_preds.append([word for word in words_pred if word != "<s>"])

    mfccs = []
    for label in y_trues:
        mfccs.append(test_data[label]['mfcc'])


    # Multiprocess
    agents = mp.cpu_count() - 1
    with mp.Pool(processes=agents) as pool:
        recognition = partial(continuous_recognition, unigram_dict, bigram_dict, phoneme_dict, hmm)
        y_preds = pool.map(recognition, mfccs)

    result_dict = {"y_trues": y_trues, "y_preds": y_preds}
    np.save("results.npy", result_dict)

    result_dict = np.load("results.npy").item()
    print(len(result_dict["y_trues"]), len(result_dict["y_preds"]))

if __name__ == "__main__":
    main()
