# Statistical Machine Translation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sayarghoshroy/Statistical-Machine-Translation/blob/master/SMT_English_to_Hindi.ipynb)

---

### A simple SMT system for English to Hindi

- trained using parallel sentences in the train-set

- further fine-tuned using the development set

- evaluated on a portion of the available test set

<br>

## Word Translation Task

Probabilities computed using standard IBM Model 1 Algorithm implemented from scratch 

### Models


1. Standard Expectation Maximization Algorithm trained for 128 iterations


2. EM till convergence - less than a 0.005 change in probabilities


3. Considering 1000 most frequently occuring Hindi and English words - till convergence

<br>

> Alignments for the first words of particular English sentences evaluated using the trained translation probabilities of IBM Model 1 following ‘HMM based Word Alignment in Statistical Translation’, Vogel et al., 1996

> First order transition probabilities initialized using the technique outlined in ‘Word Alignment for Statistical Machine Translation Using Hidden Markov Models’ by Anahita Mansouri Bigvand

> Instead of capturing the absolute positions for word alignments, only the relative positions i.e the jump widths are taken into consideration

> Language model for Hindi with the intuition of generating coherent Hindi text: Bigram model with Laplace smoothing and backoff

> Greedily generate the translated sentence based on the above components

<br>

## Evaluation

Since the English-Hindi parallel sentences had no gold-standard annotations for alignment, metrics such as alignment precision, alignment recall and alignment error rate could not be computed. Therefore, BLEU (BLEU: a Method for Automatic Evaluation of Machine Translation, Papineni et al., 2002) scores using various smoothing methods were considered.

METEOR for Hindi would be a much better metric as is stated in these two papers:

- Assessing the Quality of MT Systems for Hindi to English Translation, Kalyani et al., 2014

- METEOR-Hindi : Automatic MT Evaluation Metric for Hindi as a Target Language, Gupta et al., 2010

These papers prove that particularly for Hindi, the METEOR metric correlates more strongly to human evaluations. However, there were no readily available implementations of METEOR-Hindi.

The trained translation models for the three settings can be found [here](https://drive.google.com/drive/folders/1Ccgy3414A4idj7VQVM64mlEutw9RhgF_?usp=sharing).

All outputs for each of the three trained models can be viewed [here](https://docs.google.com/document/d/1h0GiTy81rBiaZKcpDGsFJGUwA_Cj2hA6J2-QeBoFik8/edit?usp=sharing).

<br>

## Results

The third model performed better than the first two in terms of BLEU. It took only 24 iterations to reach convergence while the second model took 48. However, qualitatively looking at the translations, the first model did a somewhat better job. The average BLEU scores for the three models using six different smoothing methods:


| Smoothing |   1   |   2   |   3   |   4   |   5   |   7   |
|---------|---------|---------|---------|---------|---------|---------|
| Model 1 | 0.03423 | 0.16817 | 0.06805 | 0.14388 | 0.06571 | 0.18977 |
| Model 2 |0.034231 | 0.16817 | 0.06805 |0.143888 | 0.06571 | 0.18977 |
| Model 3 | 0.04127 | 0.19174 |0.082069 | 0.158630 |0.077351 |0.205563 |

Refer to the notebook for implementation details.

<br>

---