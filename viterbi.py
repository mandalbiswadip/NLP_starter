from time import time
import nltk
import numpy as np
import pandas as pd



A = np.array([[0.38, 0.32, 0.04, 0.  , 0.  , 0.11, 0.01, 0.14, 0.  ],
       [0.  , 0.58, 0.  , 0.  , 0.  , 0.42, 0.  , 0.  , 0.  ],
       [0.  , 0.07, 0.  , 0.05, 0.32, 0.  , 0.  , 0.25, 0.11],
       [0.07, 0.08, 0.  , 0.  , 0.  , 0.  , 0.2 , 0.61, 0.13],
       [0.2 , 0.3 , 0.  , 0.  , 0.  , 0.24, 0.15, 0.11, 0.  ],
       [0.18, 0.22, 0.  , 0.  , 0.2 , 0.07, 0.16, 0.11, 0.06],
       [0.  , 0.88, 0.  , 0.  , 0.  , 0.12, 0.  , 0.  , 0.  ],
       [0.  , 0.  , 0.  , 0.22, 0.28, 0.39, 0.1 , 0.  , 0.01],
       [0.57, 0.28, 0.  , 0.  , 0.  , 0.15, 0.  , 0.  , 0.  ]])


B = np.array([[1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0.69, 1., 0.88, 1., 0., 0., 0.01, 0.66, 0.38, 0., 0., 0.],
       [0., 0., 0.31, 0., 0.12, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0.99, 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.34, 0.62, 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.]])





class ViterbiAlgo(object):


    def __init__(self, start_token = "<s>", end_token="</s>") -> None:
        self.start_token = start_token
        self.pos_rows = [start_token, 'DT', 'NN', 'VB', 'VBZ', 'VBN', 'JJ', 'RB', 'IN']            # 9 tags
        self.pos_cols = ['DT', 'NN', 'VB', 'VBZ', 'VBN', 'JJ', 'RB', 'IN', end_token]
        self.pos_tags = self.pos_rows[1:]
        self.transition = A        #(9* 9)


        self.observe_likelihood_row = self.pos_tags                  # 14
        self.word_list = ['a', 'the', 'chair', 'chairman', 'board', 'road', 'is', 'was', 'found', 'middle', 'bold', 'completely', 'in', 'of']                  # 14
        self.observe_likelihood = B  # (8*14)


    def estimate_tags(self, sentence, tokenizer=None, lower=True):
        """estimate pos tags using the viterbi algorithm

        Args:
            sentence (str): input sentence
            tokenizer :  Defaults to None.
            lower (bool, optional): wheather to lowercase input. Defaults to True.
        """
        if lower:
            sentence = sentence.lower()

        words = None
        if tokenizer:
            words = tokenizer.tokenze(sentence)
        else:
            words = nltk.word_tokenize(sentence)

        T = len(words)
        N = len(self.pos_tags)


        # the viterbi table
        viterbi = np.zeros((N, T))
        backpointer = np.zeros((N,T))

        for index, state in enumerate(self.pos_tags):
            state_observation_likelihood = self.observe_likelihood[self.observe_likelihood_row.index(state), self.word_list.index(words[0])]
            start_token_row_index = self.pos_rows.index(self.start_token)
            state_column_index = self.pos_cols.index(state)
            viterbi[index, 0] = self.transition[start_token_row_index, state_column_index] * state_observation_likelihood

        
        for timestep in range(1,T):
        # at each word    
            
            # at each pos tag
            for index, state in enumerate(self.pos_tags):
                
                
                max_prob_state_index = None
                max_prob = 0

                state_observation_likelihood = self.observe_likelihood[self.observe_likelihood_row.index(state), self.word_list.index(words[timestep])]

                for prev_index, prev_state in enumerate(self.pos_tags):
                    prob = viterbi[prev_index, timestep-1] * self.transition[self.pos_rows.index(prev_state), self.pos_cols.index(state)] * state_observation_likelihood
                    if prob > max_prob:
                        max_prob_state_index = prev_index
                        max_prob = prob
                viterbi[index, timestep] = max_prob
                backpointer[index, timestep] = max_prob_state_index

        return viterbi, backpointer


def get_pos_tags(sentence, vit_model):
    """

    Args:
        sentence (_type_): _description_

    Returns:
        list: list of (word, tag)
    """
    print("SENTENCE: ", sentence)
    table, back_point = vit_model.estimate_tags(sentence)
    table_df = pd.DataFrame(table)
    table_df.columns = sentence.split()
    table_df.index = vit_model.pos_tags
    print("viterbi table\n", table_df)

    backpointer_df = pd.DataFrame(back_point)
    backpointer_df.columns = sentence.split()
    backpointer_df.index = vit_model.pos_tags
    print("Backpointer table\n", backpointer_df)
    

    tags = []
    index = np.argmax(table[:, -1])
    tags.append(vit_model.pos_tags[index])
    for i in range(len(sentence.split()) -1):
        index = back_point[:, - 1 - i][int(index)]
        tags.append(vit_model.pos_tags[int(index)])
    return list(zip(sentence.split(), tags[::-1]))





if __name__=="__main__":
    vit = ViterbiAlgo()

    s1 = "The chairman of the board is completely bold".lower()

    s2 = "A chair was found in the middle of the road".lower()

    print(get_pos_tags(s1, vit))
    print("\n\n")
    print(get_pos_tags(s2, vit))

    
    