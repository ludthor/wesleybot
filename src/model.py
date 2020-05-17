from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam

import numpy as np
from random import randint

class Model:

    def __init__(self, max_length, chars, c2i, i2c, corpus):
        self.max_length = max_length
        self.chars = chars
        self.chars_len = len(chars)
        self.temperature = 1.0
        self.c2i = c2i
        self.i2c = i2c
        self.corpus = corpus
        self.generated_text = []

    def configure_model(self):
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(self.max_length,
                                              self.chars_len)))
        self.model.add(Dense(self.chars_len, activation='softmax'))

        optimizer = Adam(learning_rate=0.01)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer)

        print(self.model.summary())

    def fit_model(self, x, y, batch_size, epochs):
        print("\nT R A I N I N G - M O D E L\n")
        print_callback = LambdaCallback(on_epoch_end=self.on_epoch_end)
        self.model.fit(x, y,
                       batch_size=batch_size,
                       epochs=epochs,
                       callbacks=[print_callback])

    """
    Taken as it from the Jeff Heaton github repository

    The LSTM will produce new text character by character. 
    We will need to sample the correct letter from the LSTM predictions each time. 
    The sample function accepts the following two parameters:

    preds - The output neurons.
    temperature - 1.0 is the most conservative, 
                0.0 is the most confident (willing to make spelling and other errors).
    The sample function below is essentially performing a softmax on the neural network predictions.
    This causes each output neuron to become a probability of its particular letter.
    """
    def sample(self, preds):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / self.temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    """
    Callback function called after every epoch
    """
    def on_epoch_end(self, epoch, _):
        print("\nE N D - O F - E P O C H\n")

        start = randint(0, len(self.corpus) - 1 - self.max_length)

        generated = ''
        sentence = self.corpus[start: start + self.max_length]

        for i in range(280):
            # Init the array of possibilities with 0
            x_pred = np.zeros((1, self.max_length, self.chars_len))
            # Populate the array before sending to the model 
            # with the initial text
            for t, char in enumerate(sentence):
                x_pred[0, t, self.c2i[char]] = 1
            
            preds = self.model.predict(x_pred, verbose=0)[0]
            # Take the prediction (An array of probabilities)
            # And get the index of the MOST PROBABLE CHARACTER
            # Using the softmax defined previously
            next_index = self.sample(preds)
            # Map the gathered index into the corresponding char
            # Using the original dictionary index2char
            next_char = self.i2c[next_index]

            # Concatenate chars
            generated += next_char
            sentence = sentence[1:] + next_char

        # generated stores the final prediction
        # print and save
        generated += sentence
        print("#"*64)
        print("Epoch: ", epoch)
        print("Text: \n", generated)
        print("#"*64)
        self.generated_text.append(generated)

    def save_generated(self):
        print("S A V I N G - G E N E R A T E D")
        text_file = open("../result/predictions_by_epoch.txt", 'w')
        for i, t in enumerate(self.generated_text):

            text_file.write("\n\nEpoch:" + str(i) + "\n" + str(t))

        text_file.close()
