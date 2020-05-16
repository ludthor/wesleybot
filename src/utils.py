import os


class Utils:

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
        chars = sorted(list(set(text)))
        print("Dictionary of unique characters:\n")
        print(chars)

        c2i = dict((c, i) for i, c in enumerate(chars))
        i2c = dict((i, c) for i, c in enumerate(chars))

        return c2i, i2c
