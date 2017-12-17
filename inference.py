import pandas as pd
import numpy as np



def viterbi_isolated(hmm, start, mfcc, word, phonemes):
    """ISOLATED WORD RECOGNITION

    Args
        start (int): 0<= start < T
        mfcc (2d matrix): (time, n_features)
        word (str): words in unigram
        phonemes (list): sequence of phonemes in a given word

    Attributes
        v_path: viterbi path probability with length T. We just need the max probability at T
            and since the word is given, let's just assume that mfcc follows sequence of word phonemes.

    Returns
        t, Viterbi_path(=Max probability of Observation given word P(O|w)), infrenced phonemes
    """

    # Case1: Don't Skip sp.
    result1 = {}
    total_length = mfcc.shape[0]
    v_path = []
    phoneme_index = []
    cur_phoneme_idx = 0
    cur_state = 1  # Starts as state 1, states are 0~2 for sp, and 0~4 for rest.

    for t in range(start, total_length):
        cur_phoneme = phonemes[cur_phoneme_idx]

        if(t == start):
            # We need to consider that sp can jump to next phoneme without staying.
            log_emission_prob = hmm.emission_prob(mfcc[t], hmm.gauss_mixtures_dict(cur_phoneme, state=cur_state+1))
            viterbi_prob = log_emission_prob + np.log(hmm.initial_prob(cur_phoneme))
            v_path.append(viterbi_prob)  # Log for floating point precision
            phoneme_index.append(cur_phoneme_idx)
            continue

        # PHONEME TRANSITION
        # Find the maximum for the next possible phoneme
        # But the next phoneme can only be found from transition marix.
        # If still inside the matrix, cur_phoneme_idx remains the same.
        # Find the viterbi value with from all possible next states
        # NEXT STATE
        next_possible_states = np.where(hmm.transition_prob(cur_phoneme)[cur_state] > 0)[0]
        viterbi_probs = np.zeros(shape=hmm.n_states(cur_phoneme))

        for next_state in next_possible_states:
            is_leaving = (next_state + 1 == hmm.states(cur_phoneme)[-1] + 1)
            if cur_phoneme != 'sp' and is_leaving:  
                if cur_phoneme == 'sil':
                    log_emission_prob = 0
                else:
                    next_phoneme = phonemes[cur_phoneme_idx + 1]
                    log_emission_prob = hmm.emission_prob(mfcc[t], hmm.gauss_mixtures_dict(next_phoneme, state=2))
            elif cur_phoneme == 'sp' and is_leaving:
                # We need the next_phoneme from next word...... Shit..
                # log_emission_prob = hmm.emission_prob(mfcc[t], hmm.gauss_mixtures_dict(next_phoneme, state=2))
                log_emission_prob = 0
            else:
                log_emission_prob = hmm.emission_prob(mfcc[t], hmm.gauss_mixtures_dict(cur_phoneme, state=next_state+1))

            transition_prob = hmm.transition_prob(cur_phoneme)[cur_state][next_state]
            viterbi_probs[next_state] = v_path[t-start-1] + log_emission_prob + np.log(transition_prob)
            # sp is optional. So we should consider cases where we skip sp.
            # next_state + 1 is the natural number representation of state ( 1,...,5 )

            # Case where we visit to 'sp'
            # If exiting and next phoneme is 'sp' --> Add probability of going into sp
            if cur_phoneme != 'sp' and cur_phoneme !='sil' and is_leaving and phonemes[cur_phoneme_idx+1] == 'sp':
                viterbi_probs[next_state] += np.log(hmm.transition_prob('sp')[0][1])

        # Update cur_state to state with maximum probability.
        max_prob = max([viterbi_probs[next_state] for next_state in next_possible_states])
        cur_state = np.where(viterbi_probs == max_prob)[0][0]
        v_path.append(max_prob)
        phoneme_index.append(cur_phoneme_idx)

        # sp is optional
        exit_state = hmm.transition_prob(cur_phoneme).shape[1] - 1
        if cur_state == exit_state:
            cur_phoneme_idx += 1
            cur_state = 1

            if cur_phoneme_idx == len(phonemes) or t == total_length - 1:
                result1 = {'t': t, 'v_path': v_path, 'phoneme_path': [phonemes[idx] for idx in phoneme_index]}
                break;
        if t == total_length - 1:
            result1 = {'t': t, 'v_path': v_path, 'phoneme_path': [phonemes[idx] for idx in phoneme_index]}

    # Case2: skip sp.
    result2 = {}
    total_length = mfcc.shape[0]
    v_path = []
    phoneme_index = []
    cur_phoneme_idx = 0
    cur_state = 1  # Starts as state 1, states are 0~2 for sp, and 0~4 for rest.

    for t in range(start, total_length):
        cur_phoneme = phonemes[cur_phoneme_idx]
        is_skip = False

        if(t == start):
            log_emission_prob = hmm.emission_prob(mfcc[t], hmm.gauss_mixtures_dict(cur_phoneme, state=cur_state+1))
            viterbi_prob = log_emission_prob + np.log(hmm.initial_prob(cur_phoneme))
            v_path.append(viterbi_prob)  # Log for floating point precision
            phoneme_index.append(cur_phoneme_idx)
            continue

        next_possible_states = np.where(hmm.transition_prob(cur_phoneme)[cur_state] > 0)[0]
        viterbi_probs = np.zeros(shape=hmm.n_states(cur_phoneme))

        for next_state in next_possible_states:
            is_leaving = (next_state + 1 == hmm.states(cur_phoneme)[-1] + 1)
            if is_leaving and cur_phoneme == 'sil':
                log_emission_prob = 0
            elif is_leaving and not phonemes[cur_phoneme_idx+1] == 'sp':
                next_phoneme = phonemes[cur_phoneme_idx + 1]
                log_emission_prob = hmm.emission_prob(mfcc[t], hmm.gauss_mixtures_dict(next_phoneme, state=2))
            elif is_leaving and phonemes[cur_phoneme_idx+1] == 'sp':
                # We need the next_phoneme from next word...... Shit..
                # log_emission_prob = hmm.emission_prob(mfcc[t], hmm.gauss_mixtures_dict(next_phoneme, state=2))
                log_emission_prob = 0  # Should add the probability later at bigram.
            else:
                # In phoneme state transition
                log_emission_prob = hmm.emission_prob(mfcc[t], hmm.gauss_mixtures_dict(cur_phoneme, state=next_state+1))

            transition_prob = hmm.transition_prob(cur_phoneme)[cur_state][next_state]
            viterbi_probs[next_state] = v_path[t-start-1] + log_emission_prob + np.log(transition_prob)

            # Case where we skip 'sp'
            # If exiting and next phoneme is 'sp' --> Add probability of skipping sp
            if cur_phoneme != 'sp' and cur_phoneme != 'sil' and is_leaving and phonemes[cur_phoneme_idx+1] == 'sp':
                viterbi_probs[next_state] += np.log(hmm.transition_prob('sp')[0][2])
                is_skip = True

        # Update cur_state to state with maximum probability.
        max_prob = max([viterbi_probs[next_state] for next_state in next_possible_states])
        cur_state = np.where(viterbi_probs == max_prob)[0][0]
        v_path.append(max_prob)
        phoneme_index.append(cur_phoneme_idx)

        # We need to consider that sp can jump to next phoneme without staying.
        # sp is optional
        exit_state = hmm.transition_prob(cur_phoneme).shape[1] - 1
        if cur_state == exit_state or is_skip:
            cur_phoneme_idx += 1
            cur_state = 1
            if is_skip or cur_phoneme_idx == len(phonemes) or t == total_length-1:
                result2 = {'t': t, 'v_path': v_path, 'phoneme_path': [phonemes[idx] for idx in phoneme_index]}
                break;
        if t == total_length -1:
            result2 = {'t': t, 'v_path': v_path, 'phoneme_path': [phonemes[idx] for idx in phoneme_index]}

    if result1['v_path'][-1]  > result2['v_path'][-1]:
        return result1
    else:
        return result2


def continuous_recognition(unigram_dict, bigram_dict, phoneme_dict, hmm, mfcc):
    """UTTERANCE MODEL
    Returns estimated sequence of words from given mfcc.
    """

    total_length = mfcc.shape[0]
    word_path = []
    word_list = list(phoneme_dict.keys())
    viterbi_path = []

    t = 0
    while True:
        # initial
        if t == 0:
            viterbi_prob = np.zeros(shape=len(word_list)+1)  # 2 zeros
            viterbi_t = np.zeros(shape=len(word_list)+1)    # argmax_word P(word) * P(O_1|word)
            # Find initial seq
            for idx, word in enumerate(word_list):
                if word == 'zero':
                    # First zero
                    res = viterbi_isolated(hmm, t, mfcc, word, phoneme_dict[word][0])
                    p_obs_given_word = res['v_path'][-1]
                    viterbi_t[idx] = res['t']
                    viterbi_prob[idx] = np.log(unigram_dict[word]) + p_obs_given_word

                    # Second zero ( Note that there are 2 types of zeros )
                    res = viterbi_isolated(hmm, t, mfcc, word, phoneme_dict[word][1])
                    p_obs_given_word = res['v_path'][-1]
                    viterbi_t[idx+1] = res['t']
                    viterbi_prob[idx+1] = np.log(unigram_dict[word]) + p_obs_given_word
                else:
                    res = viterbi_isolated(hmm, t, mfcc, word, phoneme_dict[word])
                    p_obs_given_word = res['v_path'][-1]
                    viterbi_t[idx] = res['t']
                    viterbi_prob[idx] = np.log(unigram_dict[word]) + p_obs_given_word

            word_idx = np.argmax(viterbi_prob)
            t = int(viterbi_t[word_idx])
            word_path.append(word_list[word_idx if word_idx != len(word_list) else word_idx-1] )
            viterbi_path.append(max(viterbi_prob))
            continue

        viterbi_prob = np.zeros(shape=len(word_list)+1)  # 2 zeros
        viterbi_t = np.zeros(shape=len(word_list)+1)

        for idx, word in enumerate(word_list):
            word_before = word_path[-1]
            if word == 'zero':
                # First zero
                res = viterbi_isolated(hmm, t, mfcc, word, phoneme_dict[word][0])
                p_obs_given_word = res['v_path'][-1]
                viterbi_t[idx] = res['t']

                if not (word_before in bigram_dict.keys() and word in bigram_dict[word_before].keys()):
                    viterbi_prob[idx] = -1e30
                else:
                    viterbi_prob[idx] = viterbi_path[-1] + np.log(bigram_dict[word_before][word]) + p_obs_given_word

                # Second zero
                res = viterbi_isolated(hmm, t, mfcc, word, phoneme_dict[word][1])
                p_obs_given_word = res['v_path'][-1]
                viterbi_t[idx+1] = res['t']

                if not (word_before in bigram_dict.keys() and word in bigram_dict[word_before].keys()):
                    viterbi_prob[idx+1] = -1e30
                else:
                    viterbi_prob[idx+1] = viterbi_path[-1] + np.log(bigram_dict[word_before][word]) + p_obs_given_word
            else:
                res = viterbi_isolated(hmm, t, mfcc, word, phoneme_dict[word])
                p_obs_given_word = res['v_path'][-1]
                viterbi_t[idx] = res['t']

                if not (word_before in bigram_dict.keys() and word in bigram_dict[word_before].keys()):
                    viterbi_prob[idx] = -1e30
                else:
                    viterbi_prob[idx] = viterbi_path[-1] + np.log(bigram_dict[word_before][word]) + p_obs_given_word

        word_idx = np.argmax(viterbi_prob)
        t = int(viterbi_t[word_idx])
        word_path.append(word_list[word_idx if word_idx != len(word_list) else word_idx-1] )

        if(t == total_length-1):
            break;

    return word_path
