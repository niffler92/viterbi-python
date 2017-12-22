import numpy as np
from scipy.stats import multivariate_normal

from preprocess import get_hmm_dict

class HMM:
    """
    A class with api for needed constants and matrices.
    """
    def __init__(self, hmm_dict):
        self.hmm_dict = hmm_dict
        self.phonemes = list(self.hmm_dict.keys())

    def n_states(self, phoneme):
        return int(self.hmm_dict[phoneme]['<NUMSTATES>'])

    def transition_prob(self, phoneme):
        return np.array(self.hmm_dict[phoneme]['<TRANSP>']).astype(float)

    def states(self, phoneme):
        return sorted([int(x) for x in self.hmm_dict[phoneme] if x not in ['<NUMSTATES>', '<TRANSP>']])

    def n_mixes(self, phoneme, state):
        return int(self.hmm_dict[phoneme][str(state)]['<NUMMIXES>'])

    def gauss_mixtures_dict(self, phoneme, state):
        return self.hmm_dict[phoneme][str(state)]['<MIXTURES>']

    def initial_prob(self, phoneme):
        return 1

    @staticmethod  # considering floating point precision
    def emission_prob(x, mixture_dict):
        assert len(x) == 39

        b = []
        exp_sum = 0.0
        max_b_i = 0.0
        for mix in mixture_dict.keys():
            mean = np.array(mixture_dict[mix]['<MEAN>']).astype(float)
            variance = np.array(mixture_dict[mix]['<VARIANCE>']).astype(float)
            weight = float(mixture_dict[mix]['<MIXTURE>'])
            gconst = float(mixture_dict[mix]['<GCONST>'])

            log_b_i = np.log(weight) - gconst/2 + np.sum((-0.5) * ((x-mean) ** 2) / variance)

            b.append(log_b_i)

        max_b_i = max(b)
        max_idx = np.argmax(b)

        for i in range(len(b)):
            if i != max_idx:
                diff = b[i] - max_b_i
                exp_sum += np.exp(diff)

        return max_b_i + np.log(1+exp_sum)

    @staticmethod
    def emission_prob2(x, mixture_dict):
        """This will cause floating point precision error
        """
        assert len(x) == 39

        prob = 0
        for mix in mixture_dict.keys():
            mean = np.array(mixture_dict[mix]['<MEAN>']).astype(float)
            var = np.array(mixture_dict[mix]['<VARIANCE>']).astype(float)

            mv_norm = multivariate_normal(mean, np.eye(len(var))*var)
            weight = float(mixture_dict[mix]['<MIXTURE>'])
            prob += weight * mv_norm.pdf(x)

        return prob

class StateGraph:
    def __init__(self, hmm, phoneme_dict, bigram_dict):
        self.hmm = hmm
        self.phoneme_dict_org = phoneme_dict
        self.bigram_dict = bigram_dict
        self.word_list_org = sorted(phoneme_dict.keys())

        self.word_list = self.word_list_org.copy()
        self.word_list[-1] = 'zero1'
        self.word_list.append('zero2')

        self.phoneme_dict = self.phoneme_dict_org.copy()
        self.phoneme_dict['zero1'] = self.phoneme_dict['zero'][0]
        self.phoneme_dict['zero2'] = self.phoneme_dict['zero'][1]
        self.phoneme_dict.pop('zero')


        assert len(self.word_list_org) != len(self.word_list)
        assert len(self.phoneme_dict.keys()) != len(self.phoneme_dict_org.keys())

        self.states = []
        self.word_start_idx = []
        self.word_end_idx = []

        self.set_states()
        self.set_word_start_end()

    def set_states(self):
        for word in self.word_list:
            for phoneme in self.phoneme_dict[word]:
                    for state in self.hmm.states(phoneme):
                        self.states.append("{}_{}_{}".format(word, phoneme, state))

    def set_word_start_end(self):
        for word in self.word_list:
            word_idx = []
            for state in self.states:
                if state.startswith(word):
                    word_idx.append(self.states.index(state))

            self.word_start_idx.append(min(word_idx))
            if word != "<s>":  # Because we can skip sp, the phoneme before sp can be word end
                self.word_end_idx.append(max(word_idx)-1)
            self.word_end_idx.append(max(word_idx))

    def is_connected(self, total_prev_state_idx, total_new_state_idx, prev_viterbi_prob):
        prev_state_info = self.states[total_prev_state_idx].split("_")
        new_state_info = self.states[total_new_state_idx].split("_")

        # e.g. prev_state = eight_ey_3
        prev_word = prev_state_info[0]  # e.g. eight
        new_word = new_state_info[0]
        prev_phoneme = prev_state_info[1]  # e.g. ey
        new_phoneme = new_state_info[1]
        prev_state_idx = int(prev_state_info[2]) - 1  # e.g. <2, 3, 4> --> <1, 2, 3> for indexing
        new_state_idx = int(new_state_info[2]) - 1

        prev_states_idx = np.where(prev_viterbi_prob > -1e200)[0]
        if total_prev_state_idx not in prev_states_idx:
            return False, 0

        # We have to check availability in word, phoneme, state
        if prev_word == new_word:
            if total_prev_state_idx in self.word_end_idx and total_new_state_idx in self.word_start_idx:
                # 아래는 코드 중복이라서 refactoring 요함
                prev_word_org = 'zero' if prev_word.startswith('zero') else prev_word
                new_word_org = 'zero' if new_word.startswith('zero') else new_word
                if prev_word_org in self.bigram_dict.keys() and new_word_org in self.bigram_dict[prev_word_org].keys():
                    # We need to check if prev_phoneme is "sp" or the phoneme before "sp".
                    possible_exit_phoneme = []
                    last_phoneme = self.phoneme_dict[prev_word][-1]
                    possible_exit_phoneme.append(last_phoneme)  # Last phoneme
                    if last_phoneme == "sp":
                        assert len(self.phoneme_dict[prev_word]) > 1
                        possible_exit_phoneme.append(self.phoneme_dict[prev_word][-2])

                    possible_start_phoneme = self.phoneme_dict[new_word][0]
                    if prev_phoneme in possible_exit_phoneme and new_phoneme in possible_start_phoneme:
                        possible_exit_state = np.where(self.hmm.transition_prob(prev_phoneme)[:, -1] > 0)[0]
                        possible_start_state = 1
                        if prev_phoneme == "sp":
                            if prev_state_idx in possible_exit_state and new_state_idx == possible_start_state:
                                transition_prob = self.hmm.transition_prob(prev_phoneme)[prev_state_idx][-1]
                                bigram_prob = self.bigram_dict[prev_word_org][new_word_org]
                                assert transition_prob > 0

                                return True, np.log(transition_prob) + np.log(bigram_prob)
                        else:
                            if prev_state_idx in possible_exit_state and new_state_idx == possible_start_state:
                                transition_prob = self.hmm.transition_prob(prev_phoneme)[prev_state_idx][-1]
                                prob_skip_sp = self.hmm.transition_prob("sp")[0][-1]
                                bigram_prob = self.bigram_dict[prev_word_org][new_word_org]
                                assert transition_prob > 0
                                assert prob_skip_sp > 0

                                return True, np.log(transition_prob) + np.log(prob_skip_sp) + np.log(bigram_prob)



            if prev_phoneme == new_phoneme:  # In same phoneme
                possible_new_states = np.where(self.hmm.transition_prob(prev_phoneme)[prev_state_idx] > 0)[0]
                if new_state_idx in possible_new_states:
                    if np.abs(total_prev_state_idx - total_new_state_idx) > 3:  # To avoid two same phonemes in one word. e.g. nine has n, n
                        return False, 0
                    transition_prob = self.hmm.transition_prob(prev_phoneme)[prev_state_idx][new_state_idx]
                    assert transition_prob > 0

                    return True, np.log(transition_prob)
            else:  # phoneme transition
                ####################### 이거 바뀌어야함. nine에서 n 두 개라 첫 번째 n의 위치가 나옴. six도..
                possible_new_phoneme_idx = self.phoneme_dict[prev_word].index(prev_phoneme) + 1
                if total_prev_state_idx == 38 or total_prev_state_idx == 81:
                    possible_new_phoneme_idx = len(self.phoneme_dict[prev_word]) - 1
                if possible_new_phoneme_idx == len(self.phoneme_dict[prev_word]):  # e.g. sp can not go to other phoneme
                    return False, 0

                possible_new_phoneme = self.phoneme_dict[prev_word][possible_new_phoneme_idx]
                possible_new_state = 1  # We can only get into state 2
                possible_exit_state = np.where(self.hmm.transition_prob(prev_phoneme)[:, -1] > 0)[0]

                if new_phoneme == "sp" and possible_new_phoneme == "sp":  # We go to sp with probability
                    if prev_state_idx == possible_exit_state and new_state_idx == possible_new_state:
                        prob_to_sp = self.hmm.transition_prob("sp")[0][1]
                        transition_prob = self.hmm.transition_prob(prev_phoneme)[prev_state_idx][-1]  # Exit probability
                        assert transition_prob > 0
                        assert prob_to_sp > 0

                        return True, np.log(transition_prob) + np.log(prob_to_sp)
                elif new_phoneme == possible_new_phoneme:  # If new phoneme is not sp.
                    if prev_state_idx == possible_exit_state and new_state_idx == possible_new_state:
                        transition_prob = self.hmm.transition_prob(prev_phoneme)[prev_state_idx][-1]
                        assert transition_prob > 0

                        return True, np.log(transition_prob)
        else:  # word transition

            if total_new_state_idx not in self.word_start_idx:
                return False, 0
            if total_prev_state_idx not in self.word_end_idx:
                return False, 0
            prev_word_org = 'zero' if prev_word.startswith('zero') else prev_word
            new_word_org = 'zero' if new_word.startswith('zero') else new_word
            if prev_word_org in self.bigram_dict.keys() and new_word_org in self.bigram_dict[prev_word_org].keys():
                # We need to check if prev_phoneme is "sp" or the phoneme before "sp".
                possible_exit_phoneme = []
                last_phoneme = self.phoneme_dict[prev_word][-1]
                possible_exit_phoneme.append(last_phoneme)  # Last phoneme
                if last_phoneme == "sp":
                    assert len(self.phoneme_dict[prev_word]) > 1
                    possible_exit_phoneme.append(self.phoneme_dict[prev_word][-2])

                possible_start_phoneme = self.phoneme_dict[new_word][0]
                if prev_phoneme in possible_exit_phoneme and new_phoneme in possible_start_phoneme:
                    possible_exit_state = np.where(self.hmm.transition_prob(prev_phoneme)[:, -1] > 0)[0]
                    possible_start_state = 1
                    if prev_phoneme == "sp":
                        if prev_state_idx in possible_exit_state and new_state_idx == possible_start_state:
                            transition_prob = self.hmm.transition_prob(prev_phoneme)[prev_state_idx][-1]
                            bigram_prob = self.bigram_dict[prev_word_org][new_word_org]
                            assert transition_prob > 0

                            return True, np.log(transition_prob) + np.log(bigram_prob)
                    else:
                        if prev_state_idx in possible_exit_state and new_state_idx == possible_start_state:
                            transition_prob = self.hmm.transition_prob(prev_phoneme)[prev_state_idx][-1]
                            prob_skip_sp = self.hmm.transition_prob("sp")[0][-1]
                            bigram_prob = self.bigram_dict[prev_word_org][new_word_org]
                            assert transition_prob > 0
                            assert prob_skip_sp > 0

                            return True, np.log(transition_prob) + np.log(prob_skip_sp) + np.log(bigram_prob)

        return False, 0


# For test
if __name__ == "__main__":
    hmm = HMM(get_hmm_dict("./data/hmm.txt"))
    mixture_dict =  hmm.gauss_mixtures_dict('f', 2)

    a = 0
    print(hmm.emission_prob([a]*39, mixture_dict), np.log(hmm.emission_prob2([a]*39, mixture_dict)))
    a = 20
    print(hmm.emission_prob([a]*39, mixture_dict), np.log(hmm.emission_prob2([a]*39, mixture_dict)))
