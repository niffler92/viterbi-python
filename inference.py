import pandas as pd
import numpy as np


def viterbi(unigram_dict, hmm, state_graph, mfcc):
    T = mfcc.shape[0]
    n_states = len(state_graph.states)
    viterbi_prob = np.zeros(shape=(mfcc.shape[0], n_states)) - 1e200  # Safety Threshold for argmax
    backpointer = np.zeros_like(viterbi_prob) - 1  # To ensure that -1 is not used.

    for t in range(T):
        if t == 0:
            # Unigram for starting probability.
            for start_idx in state_graph.word_start_idx:
                state_info = state_graph.states[start_idx].split("_")
                word = state_info[0]
                word = word[:-1] if word.startswith("zero") else word
                phoneme = state_info[1]
                state = state_info[2]

                log_emission_prob = hmm.emission_prob(mfcc[t], hmm.gauss_mixtures_dict(phoneme, state))
                viterbi_prob[t, start_idx] = np.log(unigram_dict[word]) + log_emission_prob
                backpointer[t, start_idx] = 0
        else:
            for state_idx in range(n_states):
                candidates = np.zeros_like(viterbi_prob[t-1, :]) - 1e200
                cnt = 0
                for prev_state_idx in range(n_states):
                    is_connected, log_transition_prob = state_graph.is_connected(prev_state_idx, state_idx, viterbi_prob[t-1])
                    if is_connected:
                        cnt += 1
                        state_info = state_graph.states[state_idx].split("_")
                        phoneme = state_info[1]
                        state = state_info[2]
                        log_emission_prob = hmm.emission_prob(mfcc[t], hmm.gauss_mixtures_dict(phoneme, state))
                        candidates[prev_state_idx] = viterbi_prob[t-1, prev_state_idx] + log_emission_prob + log_transition_prob

                if cnt > 0:
                    viterbi_prob[t, state_idx] = max(candidates)
                    backpointer[t, state_idx] = np.argmax(candidates)
                else:
                    viterbi_prob[t, state_idx] = -1e200
                    backpointer[t, state_idx] = -1

    # EXTRACT WORD
    bp_indexes = []
    backpointer_idx = int(np.argmax(viterbi_prob[-1, :]))
    bp_indexes.append(backpointer_idx)
    for t in reversed(range(T-1)):
        backpointer_idx = int(backpointer[t, backpointer_idx])
        bp_indexes.append(backpointer_idx)

    words = []
    is_start = True
    for idx in reversed(bp_indexes):
        if is_start:
            if idx in state_graph.word_start_idx:
                start_idx = idx
                is_start = False
                next_word = list(state_graph.phoneme_dict.keys())[state_graph.word_start_idx.index(idx)]
                if next_word != "<s>":
                    if next_word.startswith("zero"):
                        next_word = "zero"
                    words.append(next_word)
        if idx > start_idx:
            is_start = True

    #if len(words) > 7:
    #    print("Cutting off tails of words...: {}".format(words[7:]))
    #    words = words[:7]

    print(words)
    return words
