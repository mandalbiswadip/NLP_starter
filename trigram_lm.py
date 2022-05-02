import math
from collections import Counter, defaultdict
from copy import deepcopy
from numpy import ma
from tqdm import tqdm
from matplotlib.pyplot import get

import nltk
import pandas as pd

# 1.


def compute_ngrams(sent, n=3):
    """Compute n-grams given sentence and n. ngram tokens are separated by _"""
    words = nltk.word_tokenize(sent.lower())
    return [("{}" + "_{}"*(n-1)).format(*words[i:i+n]) for i in range(len(words) - n + 1)]


# 2. vanilla trigrams


class VanillaTrigramLM(object):
    def __init__(self, corpus) -> None:
        self.corpus = corpus

        self.track_counts = defaultdict(dict)

        # construct count table
        for line in corpus.splitlines():
            for tg in compute_ngrams(line):
                grams = tg.split("_")
                # increase count of occurance of last tokens after first two tokens
                self.track_counts["_".join(
                    grams[:2])][grams[-1]] = self.track_counts["_".join(grams[:2])].get(grams[-1], 0) + 1
        # print(self.track_counts[self.track_counts.keys()[0]])

        self.track_prob = defaultdict(dict)

        # construct probablity table
        for bigram, bigram_count_dict in self.track_counts.items():
            total_count = sum(bigram_count_dict.values())
            for token, count in bigram_count_dict.items():
                self.track_prob[bigram][token] = count / total_count


    def get_trigram_prob(self, trigram):
        ngrams = trigram.split("_")
        bigram, last_token = "_".join(ngrams[:2]), ngrams[-1]
        return self.track_prob[bigram].get(last_token, 0)

# 3. laplace smoothing

class TrigramLaplaceSmoothingLM(object):
    """Trigram language modelling with laplace as smoothing
    """

    def __init__(self, corpus) -> None:
        self.corpus = corpus

        self.vocabulary = self.get_unique_words(corpus)

        self.track_smooth_counts = defaultdict(dict)

        print("Constructing count table")
        # construct count table. Note smoothing count is added in the next step
        for line in corpus.splitlines():
            for tg in compute_ngrams(line):
                grams = tg.split("_")
                # increase count of occurance of last tokens after first two tokens
                self.track_smooth_counts["_".join(
                    grams[:2])][grams[-1]] = self.track_smooth_counts["_".join(grams[:2])].get(grams[-1], 0) + 1
        # print(self.track_counts[self.track_counts.keys()[0]])

        # the original raw counts
        self.track_raw_counts = {bigram: sum(bigram_dict.values()) for bigram, bigram_dict in self.track_smooth_counts.items()}

        self.track_prob = defaultdict(dict)

        print("Constructing probablity table")
        # construct probablity table
        for bigram, bigram_count_dict in self.track_smooth_counts.items():
            
            # C(w(n-1))
            total_count = self.track_raw_counts[bigram]
            
            for unique_word in self.vocabulary:
                self.track_smooth_counts[bigram][unique_word] = self.track_smooth_counts[bigram].get(unique_word, 0) + 1
                self.track_prob[bigram][unique_word] = (self.track_smooth_counts[bigram][unique_word] ) / (total_count + len(self.vocabulary))
        
        # to save memory
        del self.track_smooth_counts

        print("Constructing reconstituted counts")
        # re-constituted counts
        self.track_reconstituted_count = defaultdict(dict)

        c = 0
        for bigram, bigram_prob_dict in self.track_prob.items():
            # C(w(n-1))
            total_count = self.track_raw_counts[bigram]
            for unique_word in self.vocabulary:
                self.track_reconstituted_count[bigram][unique_word] = self.track_prob[bigram][unique_word] * total_count
            c += 1
            if c > 1000:
                break

    @staticmethod
    def get_unique_words(corpus):
        words = []
        for line in corpus.splitlines():
            line = nltk.word_tokenize(line.lower())
            for word in line:
                if word not in words:
                    words.append(word)
        return words

    def get_trigram_prob(self, trigram):
        ngrams = trigram.split("_")
        bigram, last_token = "_".join(ngrams[:2]), ngrams[-1]
        return self.track_prob[bigram].get(last_token, 0)


# 4. Katz back off

class TrigramKatzBackLM(object):
    """Trigram language modelling with katz back off as smoothing
    """

    def __init__(self, corpus) -> None:
        self.corpus = corpus

        self.vocabulary = self.get_unique_words(corpus)
        self.trigram_compute_count, self.unigram_compute_count = 0, 0


        self.track_trigram_counts = defaultdict(dict)

        # construct count table
        for line in corpus.splitlines():
            for tg in compute_ngrams(line):
                grams = tg.split("_")
                # increase count of occurance of last tokens after first two tokens
                self.track_trigram_counts["_".join(
                    grams[:2])][grams[-1]] = self.track_trigram_counts["_".join(grams[:2])].get(grams[-1], 0) + 1
        # print(self.track_counts[self.track_counts.keys()[0]])


        # bigram table needed for katz back off. The discounted probablities are estimated only when needed
        self.track_bigram_counts = defaultdict(dict)

        # construct count table
        for line in corpus.splitlines():
            for bg in compute_ngrams(line, n=2):
                grams = bg.split("_")
                # increase count of occurance of last tokens after first two tokens
                self.track_bigram_counts[grams[0]][grams[-1]
                                                   ] = self.track_bigram_counts[grams[0]].get(grams[-1], 0) + 1
        # print(self.track_counts[self.track_counts.keys()[0]])

        # unigram table needed for katz back off. The discounted probablities are estimated only when needed
        # by default count is 1 (to count for unseen words)
        self.track_unigram_counts = dict()

        # construct count table
        for line in corpus.splitlines():
            for bg in compute_ngrams(line, n=1):
                # increase count of occurance of last tokens after first two tokens
                self.track_unigram_counts[bg] = self.track_unigram_counts.get(
                    bg, 0) + 1
        # print(self.track_counts[self.track_counts.keys()[0]])

        self.track_discounted_prob = defaultdict(dict)

        # construct probablity table
        for bigram, bigram_count_dict in tqdm(self.track_trigram_counts.items()):
            total_count = sum(bigram_count_dict.values())

            for token in self.vocabulary:

                if token in bigram_count_dict:
                    count = bigram_count_dict[token]
                    self.track_discounted_prob[bigram][token] = (
                        count + 1) / (total_count + len(self.vocabulary))
                    self.trigram_compute_count += 1

                else:
                    # when trigram count is 0, use the katz back off strategy
                    # NOTE: only counting the unigram smooth probablity compute count as 
                    # actually computing the probablity takes a huge memory and time
                    # we will only compute the probablity only when input sentence 
                    # comes using the `get_katz_back_probablity` function
                    trigram = bigram + "_" + token
                    ind = self.get_compute_indicator(trigram)
                    
                    if ind == 1:
                        self.unigram_compute_count += 1


    def get_compute_indicator(self, trigram):

        trigram_indicator, bigram_indicator, unigram_indicator = 3, 2, 1

        ngrams = trigram.split("_")
        bigram, last_token = "_".join(ngrams[:2]), ngrams[-1]

        # try trigrams probablity
        if bigram in self.track_discounted_prob:
            bigram_prob_dict = self.track_discounted_prob[bigram]
            if last_token in bigram_prob_dict:
                # discounted prob
                return trigram_indicator

        # back off to bigram probablity
        first_unigram, second_unigram = ngrams[1:]
        if first_unigram in self.track_bigram_counts:
            if second_unigram in self.track_bigram_counts[first_unigram]:
                return bigram_indicator

        return unigram_indicator
        

    def get_katz_back_probablity(self, trigram):
        """get the katz back off probablity for a trigram

        Args:
            trigram (str): input trigram. unigrams are separated by "_"

        Returns:
            float, int: the katz back off probablity, if the probablity is 1(unigram), 2(bigram), 3(trigram)
        """
        trigram_indicator, bigram_indicator, unigram_indicator = 3, 2, 1

        ngrams = trigram.split("_")
        bigram, last_token = "_".join(ngrams[:2]), ngrams[-1]

        # try trigrams probablity
        if bigram in self.track_discounted_prob:
            bigram_prob_dict = self.track_discounted_prob[bigram]
            if last_token in bigram_prob_dict:
                # discounted prob
                return bigram_prob_dict[last_token], trigram_indicator

        # back off to bigram probablity
        first_unigram, second_unigram = ngrams[1:]
        if first_unigram in self.track_bigram_counts:
            if second_unigram in self.track_bigram_counts[first_unigram]:
                bigram_prob_discounted = (self.track_bigram_counts[first_unigram][second_unigram] + 1) / (
                    sum(self.track_bigram_counts[first_unigram].values()) + len(self.vocabulary))
                return self.get_trigram_alpha(bigram) * bigram_prob_discounted, bigram_indicator

        # back off to unigram probablity
        unigram_prob_discounted = None

        unigram_prob_discounted = self.get_unigram_discounted_probe(
            first_unigram)
        return unigram_prob_discounted * self.get_bigram_alpha(first_unigram), unigram_indicator

    def get_trigram_alpha(self, bigram):
        """alpha(wi, wi-1)"""
        first_word, second_word = bigram.split("_")

        unigram_count_dict = self.track_trigram_counts[bigram]

        remaining_prob = 1
        
        for un, count in unigram_count_dict.items():
            # this check is kind-of trivial
            if count > 0:
                remaining_prob -= self.track_discounted_prob[bigram].get(un, 0)

        probsum = 0
        for unigram in self.track_discounted_prob[bigram]:
            probsum += self.get_bigrams_discounted_prob(
                second_word + "_" + unigram)
        denominator = (1 - probsum)
        return remaining_prob / denominator

    def get_bigram_alpha(self, word):
        """alpha(wi)"""
        remaining_prob = 1
        for unigram in self.track_bigram_counts[word]:
            remaining_prob -= self.get_bigrams_discounted_prob(
                word + "_" + unigram)

        prob_sum = 0

        for unigram in self.track_bigram_counts[word]:
            prob_sum += self.get_unigram_discounted_probe(unigram)

        return remaining_prob / (1 - prob_sum)

    @staticmethod
    def get_unique_words(corpus):
        words = []
        for line in corpus.splitlines():
            line = nltk.word_tokenize(line.lower())
            for word in line:
                if word not in words:
                    words.append(word)
        return words

    def get_bigrams_discounted_prob(self, bigram):
        ngrams = bigram.split("_")
        bigram_prob_discounted = (self.track_bigram_counts[ngrams[0]].get(ngrams[1], 0) + 1) / (
            sum(self.track_bigram_counts[ngrams[0]].values()) + len(self.vocabulary))
        return bigram_prob_discounted

    def get_unigram_discounted_probe(self, unigram):

        # when the word is unseen in the whole corpus
        if unigram not in self.track_unigram_counts:
            return 1 / (sum(self.track_unigram_counts.values()) + 1)

        unigram_prob_discounted = self.track_unigram_counts[unigram] / (
            sum(self.track_unigram_counts.values()) + 1)
        return unigram_prob_discounted

    def get_trigram_prob(self, trigram):
        return self.get_katz_back_probablity(trigram)[0]


def get_sentence_probablity(sentence, lm):
    log_probs = 0
    for trigram in compute_ngrams(sentence.lower(), n=3):
        prob = lm.get_trigram_prob(trigram)
        if prob == 0:
            return prob
        log_probs +=  math.log(  prob)
    return math.exp(log_probs)


if __name__ == "__main__":

    # UNCOMMENT PRINT STATEMENTS IF YOU WANT TO SEE INTERMEDIATE RESULTS
    with open("../corpus.txt", "r") as f:
        corpus = f.read()
    s1 = "Sales of the company to return to normalcy"
    s2 = "The new products and services contributed to increase revenue"

    # 1.
    # vanilla trigram language model
    lm = VanillaTrigramLM(corpus)

    # print count table
    # print(pd.DataFrame.from_dict(
    #     lm.track_counts, orient="index"
    #     ).fillna(0))

    # print prob table
    # print(pd.DataFrame.from_dict(
    #     lm.track_prob, orient="index"
    #     ).fillna(0))

    print("S1 probability using Vanilla trigram: ", get_sentence_probablity(s1, lm))
    print("S2 probability using Vanilla trigram: ", get_sentence_probablity(s2, lm))


    # 2. 
    lm = TrigramLaplaceSmoothingLM(corpus)

    # #print smooth count
    # print(pd.DataFrame.from_dict(
    # lm.track_smooth_counts,
    # orient="index"
    # ))

    # # print prob
    # print(pd.DataFrame.from_dict(
    # lm.track_prob,
    # orient="index"
    # ))


    # print reconstituted copunt
    # print(pd.DataFrame.from_dict(
    # lm.track_reconstituted_count,
    # orient="index"
    # ))

    # print(pd.DataFrame.from_dict(
    #     lm.track_reconstituted_count,
    #     orient="index"
    # ))
    print("S1 probability using Laplace smoothing: ", get_sentence_probablity(s1, lm))
    print("S2 probability using Laplace smoothing: ", get_sentence_probablity(s2, lm))



    3. 

    lm = TrigramKatzBackLM(corpus)

    print("S1 probability using Katz back off smoothing: ", get_sentence_probablity(s1, lm))
    print("S2 probability using Katz back off smoothing: ", get_sentence_probablity(s2, lm))



