import pickle

tokenized_stores = {'en_train': [], 'en_dev': [], 'en_test': [], 'hi_train': [], 'hi_dev': [], 'hi_test': []}

for key in tokenized_stores:
    file_name = "./Data_Files/" + str(key)[3:] + "." + str(key)[0:2]
    load = open(file_name)
    sentences = load.read().split('\n')
    
    for sentence in sentences:
        token_store = sentence.split(' ')
        tokenized_stores[key].append(token_store)

print(tokenized_stores['hi_train'][2])

train_size = len(tokenized_stores['en_train'])
dev_size = len(tokenized_stores['en_dev'])
test_size = len(tokenized_stores['en_test'])

# making the vocabulary

en_words = {}
hi_words = {}

for key in tokenized_stores:
    if str(key)[0] == 'e':
        # creating en_words
        for sentence in tokenized_stores[key]:
            for word in sentence:
                if word in en_words:
                    en_words[word] += 1
                else:
                    en_words[word] = 1
    else:
        # creating hi_words
        for sentence in tokenized_stores[key]:
            for word in sentence:
                if word in hi_words:
                    hi_words[word] += 1
                else:
                    hi_words[word] = 1
                    
en_vocab = len(en_words)
hi_vocab = len(hi_words)
print("Number of Unique Words:")
print("> English:", str(en_vocab))
print("> Hindi:", str(hi_vocab))

# creating the 't'
t = {}
# usage: t[('EN_word', 'HI_word')] = probability of EN_Word given HI_word
uniform = 1 / (en_vocab * hi_vocab)

n_iters = 0
max_iters = 25

fine_tune = 1
has_converged = False

while n_iters < max_iters and has_converged == False:
    has_converged = True
    max_change = -1

    n_iters += 1
    count = {}
    total = {}
    for index in range(train_size):
        s_total = {}
        for en_word in tokenized_stores['en_train'][index]:
            s_total[en_word] = 0
            for hi_word in tokenized_stores['hi_train'][index]:
                if (en_word, hi_word) not in t:
                    t[(en_word, hi_word)] = uniform
                s_total[en_word] += t[(en_word, hi_word)]

        for en_word in tokenized_stores['en_train'][index]:
            for hi_word in tokenized_stores['hi_train'][index]:
                if (en_word, hi_word) not in count:
                    count[(en_word, hi_word)] = 0
                count[(en_word, hi_word)] += (t[(en_word, hi_word)] / s_total[en_word])

                if hi_word not in total:
                    total[hi_word] = 0
                total[hi_word] += (t[(en_word, hi_word)] / s_total[en_word])

    # estimating the probabilities

    if fine_tune == 0:
      updated = {}
      # train for all valid word pairs s.t count(en_word, hi_word) > 0
      for index in range(train_size):
          for hi_word in tokenized_stores['hi_train'][index]:
              for en_word in tokenized_stores['en_train'][index]:
                  if (en_word, hi_word) in updated:
                      continue
                  updated[(en_word, hi_word)] = 1
                  if abs(t[(en_word, hi_word)] - count[(en_word, hi_word)] / total[hi_word]) > 0.01:
                      has_converged = False
                      max_change = max(max_change, abs(t[(en_word, hi_word)] - count[(en_word, hi_word)] / total[hi_word]))
                  t[(en_word, hi_word)] = count[(en_word, hi_word)] / total[hi_word]

    elif fine_tune == 1:
      # train it only for 1000 most frequent words in English and Hindi
      max_words = 1000
      n_hi_words = 0
      updates = 0

      for hi_word_tuples in sorted(hi_words.items(), key = lambda k:(k[1], k[0]), reverse = True):
          hi_word = hi_word_tuples[0]
          n_hi_words += 1
          if n_hi_words > max_words:
              break
          n_en_words = 0
          for en_word_tuples in sorted(en_words.items(), key = lambda k:(k[1], k[0]), reverse = True):
              en_word = en_word_tuples[0]
              n_en_words += 1
              if n_en_words > max_words:
                  break
              if (en_word, hi_word) not in count or hi_word not in total:
                  continue
                  # assume in this case: t[(en_word, hi_word)] = uniform
              else:
                  if abs(t[(en_word, hi_word)] - count[(en_word, hi_word)] / total[hi_word]) > 0.005:
                      has_converged = False
                      max_change = max(max_change, abs(t[(en_word, hi_word)] - count[(en_word, hi_word)] / total[hi_word]))
                  t[(en_word, hi_word)] = count[(en_word, hi_word)] / total[hi_word]

    print("Iteration " + str(n_iters) + " Completed, Maximum Change: " + str(max_change))

# displaying the most confident translation pairs
limit = 40
for element in sorted(t.items(), key = lambda k:(k[1], k[0]), reverse = True):
  print(element)
  limit -= 1
  if limit <= 0:
    break

# saving the translation model
file = open("IBM_model_1_translation_top_1000.pkl","wb")
pickle.dump(t,file)
file.close()

# using the model trained until convergence
pickle_in = open("/content/drive/My Drive/NLP_Translation/IBM_model_1_translation_128_iters.pkl","rb")
t = pickle.load(pickle_in)

I = {}
for index in range(train_size):
    for en_id in range(len(tokenized_stores['en_train'][index])):
        length = len(tokenized_stores['en_train'][index])
        if length not in I:
            I[length] = {} #maps the positional difference to a tuple: (sum of t's, count)
        for hi_id in range(len(tokenized_stores['hi_train'][index])):
            if (hi_id - en_id) not in I[length]:
                I[length][(hi_id - en_id)] = [t[(tokenized_stores['en_train'][index][en_id], tokenized_stores['hi_train'][index][hi_id])], 1]
            else:
                I[length][(hi_id - en_id)][0] += t[(tokenized_stores['en_train'][index][en_id], tokenized_stores['hi_train'][index][hi_id])]
                I[length][(hi_id - en_id)][1] += 1

# printing the available sentence lengths encountered during training
sentence_lengths = []
for key in I.keys():
    if key not in sentence_lengths:
        sentence_lengths.append(key)
sentence_lengths.sort()
print(sentence_lengths)

# computing the alignment probabilities
# p[I][hi_id - en_id] = p(i | i', I)

p = {}
for key in I.keys():
    p[key] = {}
    sum_val = 0
    for diff in I[key].keys():
        p[key][diff] = I[key][diff][0] / I[key][diff][1]
        sum_val += p[key][diff]
    for diff in p[key].keys():
        p[key][diff] /= sum_val

print(p[1])

for index in range(train_size):
    length_en = len(tokenized_stores['en_train'][index])
    length_hi = len(tokenized_stores['hi_train'][index])
    if length_hi - length_en > 10 and length_en == 1:
        print("Length of English Sentence:", str(length_en))
        print("Length of Hindi Sentence:", str(length_hi))
# there exists an English sentence with one token s.t the Hindi translation contains 19 tokens

# computing initial transitions
init = {}
for length in p:
    max_prob = -1
    max_jump = 0
    for key in p[length].keys():
        if p[length][key] > max_prob:
            max_prob = p[length][key]
            max_jump = key
    init[length] = max_jump

print(init)

# computing the transition probabilities for Hindi
bigrams = {}
unigrams = {}

# training on the train_set
def model(dataset_size, dataset_name):
    global bigrams
    global unigrams
    for index in range(dataset_size):
        token_A = ''
        for hi_token in tokenized_stores[dataset_name][index]:
            if hi_token not in unigrams:
                unigrams[hi_token] = 1
            else:
                unigrams[hi_token] += 1
            
            token_B = hi_token
            if (token_A, token_B) not in bigrams:
                bigrams[(token_A, token_B)] = 1
            else:
                bigrams[(token_A, token_B)] += 1
            token_A = token_B

model(train_size, 'hi_train')
model(dev_size, 'hi_dev')

bigram_count = len(bigrams)
unigram_count = len(unigrams)
print("Number of Unique Bigrams:", bigram_count)
print("Number of Unique Unigrams:", unigram_count)

from itertools import permutations
import nltk

computed_sentences = []
total_BLEU = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 7: 0}
null_BLEU_count = 0

sorted_t = sorted(t.items(), key = lambda k:(k[1], k[0]), reverse = True)

def find_translation(en_token):
    for element in sorted_t:
        if element[0][0].lower() == en_token:
            return element[0][1]
    return ""

def get_prob(seq):
    # bigram language model with laplace smoothing and backoff
    if len(seq) < 2:
        return 1
    score = 0
    token_A = ''
    for hi_token in seq:
        token_B = hi_token
        if (token_A, token_B) not in bigrams:
            if token_B not in unigrams:
                continue
            else:
                score += unigrams[token_B] / unigram_count
        else:
            score += (bigrams[(token_A, token_B)] + 1)/ (unigrams[token_A] + unigram_count)
        token_A = token_B
    return score

count = 0
for index in range(test_size):
    if len(tokenized_stores['en_test'][index]) > 8 or len(tokenized_stores['en_test'][index]) < 2:
        continue

    translated_words = []
    for en_token in tokenized_stores['en_test'][index]:
        translation = find_translation(en_token)
        if translation != "":
            translated_words.append(translation)

    perm = permutations(translated_words)

    best_seq = translated_words
    best_prob = -1

    for seq in perm:
        prob = get_prob(seq)
        if prob > best_prob:
            best_prob = prob
            best_seq = seq

    BLEU_scores = []

    # Collecting BLEU_scores with various kinds of Smoothing
    BLEU_scores.append(nltk.translate.bleu_score.sentence_bleu([tokenized_stores['hi_test'][index]], best_seq, smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1))
    BLEU_scores.append(nltk.translate.bleu_score.sentence_bleu([tokenized_stores['hi_test'][index]], best_seq, smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method2))
    BLEU_scores.append(nltk.translate.bleu_score.sentence_bleu([tokenized_stores['hi_test'][index]], best_seq, smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method3))
    BLEU_scores.append(nltk.translate.bleu_score.sentence_bleu([tokenized_stores['hi_test'][index]], best_seq, smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method4))
    BLEU_scores.append(nltk.translate.bleu_score.sentence_bleu([tokenized_stores['hi_test'][index]], best_seq, smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method5))
    BLEU_scores.append(nltk.translate.bleu_score.sentence_bleu([tokenized_stores['hi_test'][index]], best_seq, smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method7))

    for key in total_BLEU.keys():
        if key == 7:
            consider = 5
        else: consider = key - 1
        total_BLEU[key] += BLEU_scores[consider]
    
    if BLEU_scores[0] == 0:
        null_BLEU_count += 1
    
    count += 1
    print("Sentence Index: ", str(count))
    print("English Sentence:", str(tokenized_stores['en_test'][index]))
    print("Reference Hindi Sentence:", str(tokenized_stores['hi_test'][index]))
    print("Translated Sentence:", str(best_seq))
    print("Translation BLEU Scores", str(BLEU_scores))
    
    computed_sentences.append([tokenized_stores['en_test'][index], tokenized_stores['hi_test'][index], best_seq, BLEU_scores])

tested = count

# Results:
import statistics
print("Number of Samples Tested Upon: " + str(tested))
print()

print("Average BLEU Score using Various Smoothing Functions (considering all test samples)")
for key in total_BLEU:
    print("Method " + str(key) + ": " + str(total_BLEU[key] / tested))
print()
print("Average BLEU Score using Various Smoothing Functions (considering test samples with at-least one word overlap)")
for key in total_BLEU:
    print("Method " + str(key) + ": " + str(total_BLEU[key] / (tested - null_BLEU_count)))

# ^_^ Thank You