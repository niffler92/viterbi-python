# Automatic Speech Recognition with HMM
* Feature Extraction:
    * 39 MFCC features
* Acoustic Model:
    * Mixture Gaussian model for $p(o | q)$
* Lexicon/Pronunciation Model
    * HMM: what phones can follow each other
* Language Model
    * N-grams for computing $p(w_i | w_{i-1})$
* Decoder
    * Viterbi Algorithm: dynamic programming for combining all these to get word squence from speech

# Study link
* [Stanford CS224s](https://web.stanford.edu/class/cs224s/lectures/224s.17.lec3.pdf)
