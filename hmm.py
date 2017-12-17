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
        print(phoneme)
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


# For test
if __name__ == "__main__":
    hmm = HMM(get_hmm_dict("./data/hmm.txt"))
    mixture_dict =  hmm.gauss_mixtures_dict('f', 2)

    a = 0
    print(hmm.emission_prob([a]*39, mixture_dict), np.log(hmm.emission_prob2([a]*39, mixture_dict)))
    a = 20
    print(hmm.emission_prob([a]*39, mixture_dict), np.log(hmm.emission_prob2([a]*39, mixture_dict)))
