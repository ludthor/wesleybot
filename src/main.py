from reader import Reader
from utils import Utils
from random import randint
def main():
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
    utils = Utils()
    c2i, i2c = utils.create_dicts(corpus)




if __name__ == "__main__":
    main()

