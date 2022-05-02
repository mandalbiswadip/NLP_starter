from ctypes import ArgumentError
import re
from traceback import print_tb

from collections import defaultdict

def tokenize(sent):
    return [x.strip() for x in re.split(r"(\s|\.)", sent) if x.strip()]


class Node(object):

    def __init__(self, name, child1 = None, child2 = None):
        self.name = name
        self.child1 = child1
        self.child2 = child2
        self.word = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        # return f"name: {self.name}\nchild one: ({self.child1})\nchild two: ({self.child2})"
        return f"name: {self.name}"

    def get_symbol(self):
        return self.name
    
    def set_word(self, word):
        self.word = word



class Cell(object):
    """defines a cell of the CKY table

    Args:
        object (_type_): _description_
    """
    def __init__(self, i = None, j = None):
        self.i = i
        self.j = j
        self.node_list = []

    def set_cords(self, i, j):
        self.i = i
        self.j = j

    def add_node(self, node):
        if not isinstance(node, Node):
            raise TypeError(f"node should of type Node, but passed {type(node)}")
        self.node_list.append(node)

    def get_nodes(self):
        return self.node_list

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"cell at {self.i}, {self.j}"


class CKYParse(object):


    def __init__(self, grammer_list, name="cky_parser", ):
        """

        Args:
            grammer_list (list): list of tuples of CNF grammer rules. tuple can be (A, B) indicating A-> or (A, B, C) indicating A -> B C
            name (str, optional): name of the parser. Defaults to "cky_parser".
        """
        self.name = name

        self.valid_grammer = False
        if isinstance(grammer_list, list):
            if all([isinstance(x, tuple) and 2 <= len(x) for x in grammer_list]):
                self.valid_grammer = True
        if not self.valid_grammer:
            raise ValueError("grammer_list is not valid!!")

        self.grammer_list = grammer_list

    def parse_sentence(self, sentence):
        """parse the sentence and generate the CKY table
        Args:
            sentence (_type_): _description_

        Returns:
            _type_: _description_
        """
        sentence = sentence.lower()
        words = tokenize(sentence)
        table = defaultdict(lambda : defaultdict(lambda : Cell()))

        for j in range(1, len(words) + 1):

            word = words[j-1]

            for grammer in self.grammer_list:
                if len(grammer) >= 2 and word in grammer[1:]:
                    cell = table[j - 1][j]
                    cell.set_cords(j - 1, j)
                    node = Node(name = grammer[0])
                    node.set_word(word)
                    cell.add_node(node)         # no childs since this is the leaf
                    # table[j - 1][j] = cell    # no need for this as class instance is mutable


            for i in range(j-2, -1, -1):
                previous_symbols = set()

                for k in range(i+1, j):
                    B_cell = table[i][k]
                    C_cell = table[k][j]

                    for grammer in self.grammer_list:
                        if len(grammer) == 3:
                            for b_node in B_cell.get_nodes():
                                for c_node in C_cell.get_nodes():
                                    if grammer[1] == b_node.get_symbol() and grammer[2] == c_node.get_symbol():
                                        # avoid having duplicate symbols--> maybe don't do that?
                                        # if grammer[0] not in previous_symbols:
                                        previous_symbols.add(grammer[0])
                                        cell = table[i][j]
                                        cell.set_cords(i, j)
                                        node = Node(name = grammer[0], child1=b_node, child2=c_node)
                                        cell.add_node(node)
        
        return table

    @staticmethod
    def summarize_node(node, count=0):
        if node.word and not node.child1 and not node.child2:
            return f"({node.name} {node.word})"
        childs = []
        if node.child1:
            childs.append(node.child1)
        if node.child2:
            childs.append(node.child2)
        
        s = " ".join([f"({nd.name} {CKYParse.summarize_node(nd)})" for nd in childs])
        
        s = f"({node.name} {s})"
        return s

    def get_parses(self, table, l):
        root_cell = table[0][l]
        all_parses = []
        for root_node in root_cell.get_nodes():
            if root_node.get_symbol() in ["S", "ROOT"]:
                all_parses.append(self.summarize_node(root_node))
        return all_parses


if __name__ == "__main__":

    import sys
    if len(sys.argv) != 4:
        raise ArgumentError("Pass your arguments as following\npython3 Hw3_CKYparser.py <grammar_filename> <sentence_filename> <output_filename>")
    
    # LOAD GRAMMER
    grammer = None
    with open(sys.argv[1], "r") as f:
        grammer = f.readlines()
    grammer_list = [tuple([y.strip() for y in re.split(r"(\->|\s|\|)+", x) if y.strip()]) for x in grammer if x]
    
    cky_parser = CKYParse(grammer_list)

    
    # LOAD SENTENCES
    sentence_list = []
    with open(sys.argv[2], "r") as f:
        sentence_list = f.readlines()

    writer = open(sys.argv[3], "w")
    
    for sentence in sentence_list:
        print("sentence:", sentence)
        writer.write("sentence: " + sentence + "\n")
        
        print("Parses for the sentence:\n")
        writer.write("Parses for the sentence:\n")


        table = cky_parser.parse_sentence(sentence)
        parses = cky_parser.get_parses(table, len(tokenize(sentence.lower())))

        print("\n\n".join(parses))
        writer.write("\n\n".join(parses) + "\n\n")
        print("="*100)
    print(f"Storing all parses at {sys.argv[3]}")
    writer.close()

