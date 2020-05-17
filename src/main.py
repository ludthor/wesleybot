from reader import Reader
from utils import Utils
from model import Model
from random import randint

import logging
import os


def main():
    """
    Parameter configuration
    """
    # Is the total of characters used to predict the next one.
    max_length = 10
    # Is the leap of characters in the corpus taked in everytime.
    # "Move -step- characters from the first take to the second, and so on"
    step = 3
    batch_size = 128
    epochs = 60

    """
    Read and Preprocessing
    """
    reader = Reader("../corpus/corpus.txt")    
    corpus = reader.get_corpus()
    print("Lenght after load & clean: ", len(corpus))
    rand = randint(0,100)
    print("\nSample text: ", corpus[rand:rand+50])

    """
    Create dictionaries of enummerated characters
    """
    utils = Utils(max_length, step)
    c2i, i2c, chars = utils.create_dicts(corpus)

    """
    Sequence creation and preparation for the model
    """
    sentences, next_chars = utils.sequence_creator(corpus)
    rand = randint(0,100)
    print("Sample of sequence: ", sentences[rand])
    print("Sample of next char: ", next_chars[rand])

    """
    Word vectorization
    """
    x, y = utils.word_vectorization(sentences, c2i, next_chars)

    print("\nShape of vectorized sequences: ", x.shape)
    print("\nShape of vectorized prediction: ", y.shape)

    """
    Build Keras LSTM model
    """
    print("B U I L D I N G - M O D E L")
    model = Model(max_length, chars, c2i, i2c, corpus)
    model.configure_model()
    model.fit_model(x, y, batch_size, epochs)
    """
    Save the model into a file in the result folder
    """
    model.save_generated()


if __name__ == "__main__":
    
    logging.disable(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    main()

