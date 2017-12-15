# Automatic Speech Recognition with HMM
* Feature Extraction:
    * 39 MFCC features
* Acoustic Model:
    * Mixture Gaussian model for <a href="https://www.codecogs.com/eqnedit.php?latex=p(o|q)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(o|q)" title="p(o|q)" /></a>
* Lexicon/Pronunciation Model
    * HMM: what phones can follow each other
* Language Model
    * N-grams for computing <a href="https://www.codecogs.com/eqnedit.php?latex=p(w_i|w_{i-1})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(w_i|w_{i-1})" title="p(w_i|w_{i-1})" /></a>
* Decoder
    * Viterbi Algorithm: dynamic programming for combining all these to get word squence from speech

# Study link
* [Stanford CS224s](https://web.stanford.edu/class/cs224s/lectures/224s.17.lec3.pdf)
