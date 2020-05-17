import os
import numpy as np


class Utils:

    def __init__(self, max_length, step):
        self.max_length = max_length
        self.step = step

    def join_txt(self, path, type):

        data = []

        for _, _, files in os.walk(path):
            for f in files:
                with open(os.path.join(path, f)) as fp:
                    item = fp.read()
                    item += '\n'
                    data.append(item)

            with open(os.path.join(path,
                                   'corpus_'+str(type)+'.txt'), 'w') as fp:
                for d in data:
                    fp.write(d)

    def create_dicts(self, text):
        print("\n C R E A T I N G - D I C T I O N A R I E S \n")
        # set() for extract the unique values
        # sorted(list()) for sort alphabetically
        self.chars = sorted(list(set(text)))
        print("Dictionary of unique characters:\n")
        print(self.chars)

        c2i = dict((c, i) for i, c in enumerate(self.chars))
        i2c = dict((i, c) for i, c in enumerate(self.chars))

        return c2i, i2c, self.chars

    def sequence_creator(self, text):
        print("\nS E Q U E N C E - C R E A T O R\n")
        sentences = []
        next_chars = []

        # Here we are going from 0 to the last 'max_lenght' chunk,
        # hopping step by step.
        for i in range(0, len(text) - self.max_length, self.step):
            # Just take the chars from i to i + max_lenght
            sentences.append(text[i: i + self.max_length])
            # Here we just take one character, the next one from the sequence.
            next_chars.append(text[i + self.max_length])
        
        return sentences, next_chars

    def word_vectorization(self, sentences, c2i, next_chars):
        print("\nW O R D - V E C T O R I Z A T I O N\n")
        try:
            self.chars
        except NameError:
            return
        x = np.zeros((len(sentences), self.max_length,
                                      len(self.chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(self.chars)), dtype=np.bool)

        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                # For every sentence we put a 1 in the 
                # [# of sentence, character in sentence, index in dict of chars ]
                x[i, t, c2i[char]] = 1
            # for every next char we put a 1 in the
            # [# of sentence, index of the character on dict of chars ]
            y[i, c2i[next_chars[i]]] = 1
        return x, y
