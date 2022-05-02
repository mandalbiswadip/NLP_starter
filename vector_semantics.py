import math
import nltk


from collections import defaultdict
import numpy as np

def cosine_sim(a,b):
    return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))

class PPMI(object):

    def __init__(self, corpus, window=5):
        corpus = corpus.lower()
        
        # word context - dict
        self.word_context_dict = defaultdict(dict)

        for line in corpus.splitlines():
            words = nltk.word_tokenize(line)
            for index, word in enumerate(words):

                for i in range(window + 1):
                    if index - i < 0:
                        self.word_context_dict[word]["NIL"] = self.word_context_dict[word].get("NIL", 0) + 1
                    else:
                        self.word_context_dict[word][words[index - i]] = self.word_context_dict[word].get(words[index - i], 0) + 1
                    
                    if index + i >= len(words):
                        self.word_context_dict[word]["NIL"] = self.word_context_dict[word].get("NIL", 0) + 1
                    else:
                        self.word_context_dict[word][words[index + i]] = self.word_context_dict[word].get(words[index + i], 0) + 1


        self.total_count = sum([sum(val.values()) for key, val in self.word_context_dict.items()])

    def get_ppmi(self, word, context):
        word = word.lower()
        context = context.lower()

        pij = self.word_context_dict[word].get(context, 0) / self.total_count

        pi = sum(self.word_context_dict[word].values()) / self.total_count
        # you can just treat the context as another word as the word-context matrix is symmetric
        pj = sum(self.word_context_dict[context].values()) / self.total_count

        return max(math.log2(pij) - math.log2(pi) - math.log2(pj), 0)


def get_context_vector(word, word_context_dict, filter_words = None):
    if filter_words is None:
        filter_words = ["said", "of", "board"]
    
    return np.array(
        [word_context_dict[word.lower()].get(x,0) for x in filter_words]
        )


def get_word_similarity(word1, word2, word_context_dict, filter_words=None):
    return cosine_sim(get_context_vector(word1, word_context_dict, filter_words), get_context_vector(word2, word_context_dict, filter_words))




if __name__ == "__main__":
    with open("../corpus.txt", "r") as f:
        corpus = f.read()
    
    ppmi = PPMI(corpus)

    word_pairs = [
        ["chairman", "said"],
        ["chairman", "of"],
        ["company", "board"],
        ["company", "said"]
    ]

    for word_pair in word_pairs:
        score = ppmi.get_ppmi(*word_pair)
        print("PPMI of {} and {}: {}".format(
            word_pair[0], word_pair[1], round(score, 3))
            )


    word_context_dict = ppmi.word_context_dict





    word_pairs = [
        ["chairman", "company"],
        ["company", "sales"],
        ["company", "economy"]
    ]

    filter_words = ["said", "of", "board"]

    scores = []
    for word_pair in word_pairs:
        score = get_word_similarity(word_pair[0], word_pair[1], word_context_dict, filter_words)
        print("similarity of {} and {}: {}".format(
            word_pair[0], word_pair[1], round(score, 3))
            )
        scores.append(score)
    
    print("Closest word pair is {}".format(word_pairs[scores.index(max(scores))]))

