# Copyright 2019-2020, University of Freiburg
# Author: Natalie Prange <prangen@informatik.uni-freiburg.de>


import sys
import logging
import os
import getopt
import matplotlib
import datrie
import string
import numpy as np
import tensorflow as tf
# Setting the seed for numpy-generated random numbers to make sure results are
# reproduceable
# np.random.seed(2407)

import matplotlib.pyplot as plt
from enum import Enum
from time import strftime, localtime
from collections import defaultdict
from operator import itemgetter
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.models import model_from_json, load_model
from keras.backend import set_session


logging.basicConfig(format='%(asctime)s : %(message)s', datefmt="%H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

POLYAXON_EXP = False

if POLYAXON_EXP:
    from polyaxon_client.tracking import get_outputs_path
    DATA_PATH = "/data/1/prangen/data/"
    MODEL_LOAD_PATH = "data/1/prangen/model/"
    MODEL_SAVE_PATH = get_outputs_path() + "/"
    INFO_PATH = get_outputs_path() + "/"
    CHECKPOINT_SAVE_PATH = get_outputs_path() + "/"
    CHECKPOINT_LOAD_PATH = "/data/1/prangen/checkpoint/"
else:
    import global_paths as gp
    DATA_PATH = gp.LANGUAGE_MODELS_LSTM + "data/"
    MODEL_LOAD_PATH = gp.LANGUAGE_MODELS_LSTM + "model/"
    MODEL_SAVE_PATH = gp.LANGUAGE_MODELS_LSTM + "model/"
    INFO_PATH = gp.LANGUAGE_MODELS_LSTM + "info/"
    CHECKPOINT_SAVE_PATH = gp.LANGUAGE_MODELS_LSTM + "checkpoint/"
    CHECKPOINT_LOAD_PATH = ""

MIN_WORD_COUNT = 3
MAX_SEQUENCE_LEN = 30


class PredictTypes(Enum):
    NEVER = 0
    ONLY = 1
    ALSO = 2


class LM():
    def __init__(self, input_file):
        # Training data generation vars
        self.use_generator = True
        # Architecture vars
        self.embed_size = 100
        self.lstm_units = 512
        self.dropout = 0.2
        self.dense_units = 256
        # Training vars
        self.batch_size = 512
        self.num_epochs = 10
        self.val_split = 0.15

        # get the file name without path and file extension
        file_suffix = input_file.split("/")[-1]
        file_suffix = file_suffix.split(".")
        file_suffix = file_suffix[:-1] if len(file_suffix) > 1 else file_suffix
        file_suffix = '.'.join(file_suffix)

        # Load the word_dict and generate the vocab and ids files if necessary
        self.input_file = input_file
        self.ids_file = DATA_PATH+file_suffix+".ids"
        if not os.path.isfile(self.ids_file):
            self.gen_vocab(input_file, file_suffix, MIN_WORD_COUNT)
            self.get_word_dict(file_suffix)
            self.ids_file = self.gen_id_seqs(input_file, file_suffix)
        else:
            self.get_word_dict(file_suffix)

    def get_word_dict(self, file_suffix):
        with open(DATA_PATH+file_suffix+".vocab", "r",
                  encoding="latin-1") as vocab_file:
            lines = [line.strip() for line in vocab_file.readlines()]
            self.word_dict = dict([(b, a) for (a, b) in enumerate(lines)])
            self.ids_dict = dict([(a, b) for (a, b) in enumerate(lines)])

    def word_to_id(self, word):
        id = self.word_dict.get(word)
        return id if id is not None else self.word_dict.get("_UNK_")

    def id_to_word(self, id):
        word = self.ids_dict.get(id)
        return word if word is not None else "_UNK_"

    def gen_vocab(self, file_path, file_suffix, threshold):
        """Generate the vocab from the given training data
        """
        logger.info("Generate vocab for %s" % file_path)

        # Get all words from the corpus and count their occurrences
        word_counter = defaultdict(int)
        with open(file_path, "r", encoding="latin-1") as currentFile:
            for line in currentFile.readlines():
                for word in line.strip().split():
                    word_counter[word] += 1

        # Filter out words that occur less than <threshold> times in the corpus
        word_list = list()
        for word, count in sorted(word_counter.items(), key=itemgetter(1),
                                  reverse=True):
            if count >= threshold:
                word_list.append(word)

        # We need to tell LSTM the start and the end of a sentence.
        # And to deal with input sentences with variable lengths,
        # we also need padding position as 0.
        word_list = ["_PAD_", "_BOS_", "_EOS_", "_UNK_"] + word_list

        # Write the vocab to a file and create the word_dict
        with open(DATA_PATH+file_suffix+".vocab", "w",
                  encoding="latin-1") as vocab_file:
            for i, word in enumerate(word_list):
                vocab_file.write(word + "\n")

    def gen_id_seqs(self, file_path, file_suffix):
        """Generate the id sequences from the training data
        """
        logger.info("Generate id sequences for %s" % file_path)

        with open(file_path, "r", encoding="latin-1") as raw_file:
            ids_file = DATA_PATH+file_suffix+".ids"
            with open(ids_file, "w", encoding="latin-1") as current_file:
                for line in raw_file.readlines():
                    token_list = line.strip().replace("<unk>", "_UNK_").split()
                    # each sentence has the start and the end.
                    token_list = ["_BOS_"] + token_list + ["_EOS_"]
                    token_id_list = [self.word_to_id(t) for t in token_list]
                    id_string = " ".join([str(id) for id in token_id_list])
                    current_file.write("%s\n" % id_string)

        return ids_file

    def gen_training_data(self):
        """Generate the data for training the model
        """
        logger.info("Generate training data for %s" % self.ids_file)
        # create input sequences using list of tokens
        with open(self.ids_file, "r", encoding="latin-1") as file:
            input_sequences = []
            for line in file:
                token_list = [int(id) for id in line.split()]
                for i in range(len(token_list[:MAX_SEQUENCE_LEN])):
                    n_gram_sequence = token_list[:i+1]
                    input_sequences.append(n_gram_sequence)

        # pad sequences
        self.max_sequence_len = max([len(x) for x in input_sequences])
        logger.info("Max length: %d" % self.max_sequence_len)
        if self.use_generator:
            split = int(len(input_sequences) * (1-self.val_split))
            self.NUM_TRAIN_SAMPLES = len(input_sequences[:split])
            self.NUM_VAL_SAMPLES = len(input_sequences[split:])
            self.train_gen = self.batch_generator(input_sequences[:split],
                                                  self.batch_size,
                                                  mode="train")
            self.val_gen = self.batch_generator(input_sequences[split:],
                                                self.batch_size,
                                                mode="val")
        else:
            input_sequences = np.array(pad_sequences(input_sequences,
                                       maxlen=self.max_sequence_len,
                                       padding='pre'))

            # create predictors and label
            self.predictors = input_sequences[:, :-1]
            self.labels = input_sequences[:, -1]

    def batch_generator(self, input_sequences, batch_size, mode="train"):
        """Yield data batch as input for keras' fit_generator
        """
        curr_index = 0
        # Yield data batches indefinitely
        while True:
            batch_sequences = []
            while len(batch_sequences) < batch_size:
                # Fill up the batch
                batch_sequences.append(input_sequences[curr_index])
                curr_index += 1
                curr_index %= len(input_sequences)
                if curr_index == 0 and mode == "val":
                    # If we are evaluating we have to return the current batch
                    # to ensure we don't continue to fill up the batch with
                    # samples at the beginning of the file
                    break

            batch_sequences = np.array(pad_sequences(batch_sequences,
                                       maxlen=self.max_sequence_len,
                                       padding='pre'))
            # create predictors and label
            predictors = batch_sequences[:, :-1]
            labels = batch_sequences[:, -1]
            yield predictors, labels

    def create_model(self):
        """Create the LSTM model
        """
        logger.info("Train the model")
        model = Sequential()
        model.add(Embedding(len(self.word_dict), self.embed_size,
                            input_length=self.max_sequence_len-1))
        model.add(LSTM(self.lstm_units, return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.lstm_units))
        model.add(Dropout(self.dropout))
        # model.add(Dense(self.dense_units, activation='relu'))
        model.add(Dense(len(self.word_dict), activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adagrad',
                      metrics=['accuracy'])
        self.model = model

    def train_model(self):
        """Train the LSTM model
        """
        checkpoint_name = "checkpoint-{epoch:02d}-{loss:.4f}.hdf5"
        filepath = CHECKPOINT_SAVE_PATH + checkpoint_name
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,
                                     save_best_only=True, mode='min')
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                                  verbose=0, mode='auto')
        if self.use_generator:
            train_steps = self.NUM_TRAIN_SAMPLES // self.batch_size
            val_steps = self.NUM_VAL_SAMPLES // self.batch_size
            self.history = self.model.fit_generator(
                                self.train_gen,
                                steps_per_epoch=train_steps,
                                validation_data=self.val_gen,
                                validation_steps=val_steps,
                                epochs=self.num_epochs,
                                verbose=1,
                                callbacks=[earlystop, checkpoint])
        else:
            self.history = self.model.fit(self.predictors,
                                          self.labels,
                                          batch_size=self.batch_size,
                                          epochs=self.num_epochs,
                                          verbose=1,
                                          validation_split=self.val_split,
                                          callbacks=[earlystop, checkpoint])

    def predict_words(self, token_list, prefix="",
                      predict_types=PredictTypes.ALSO, max_words=20):
        # Get padded token list of input
        token_list = ["_BOS_"] + token_list
        token_list = [self.word_to_id(t) for t in token_list]
        token_list = pad_sequences([token_list],
                                   maxlen=self.max_sequence_len-1,
                                   padding='pre')

        # This is necessary for the usage in a threaded flask server
        global graph
        global session
        with graph.as_default():
            with session.as_default():
                # Get probabilities for the next word
                y_prob = self.model.predict(token_list)[0]

        if prefix:
            # Only consider words that match the prefix or are types depending
            # on predict_types
            if predict_types == PredictTypes.NEVER:
                matching_ids = self.prefix_trie.values(prefix)
            elif predict_types == PredictTypes.ONLY:
                matching_ids = self.prefix_trie.values("[")
            else:
                matching_ids = self.prefix_trie.values(prefix)
                matching_ids += self.prefix_trie.values("[")

            prob_id_arr = np.array([y_prob[matching_ids], matching_ids])
            sorted_indices = np.argsort(prob_id_arr[0], axis=-1)
            sorted_ids = prob_id_arr[1][sorted_indices].astype(int)
        else:
            # Consider probabilities for all words
            sorted_ids = np.argsort(y_prob, axis=-1)

        # Set the index for slicing the classes to length <max_words>
        if max_words:
            max_words = -max_words

        sorted_ids = sorted_ids[max_words:][::-1]
        sorted_words = [(self.id_to_word(id), y_prob[id]) for id in sorted_ids]
        return sorted_words

    def probability_for_word(self, context, word):
        """Returns the probability of a word given a context as computed by the
        language model.
        TODO: I don't actually need to compute the probabilities for all words
        but right now I don't know a more efficient way to do this with keras
        """
        token_list = ["_BOS_"] + context
        token_list = [self.word_to_id(t) for t in token_list]
        token_list = pad_sequences([token_list],
                                   maxlen=self.max_sequence_len-1,
                                   padding='pre')

        # Get probabilities for the next word
        matching_id = self.word_to_id(word)
        # This is necessary for the usage in a threaded flask server
        global graph
        global session
        with graph.as_default():
            with session.as_default():
                y_prob = self.model.predict(token_list)[0]
        return y_prob[matching_id]

    def probability_for_context(self, context):
        """Returns the probability for a given context
        = product of the probabilities of all words in the context given their
        context
        """
        if len(context) == 0:
            return 1
        return self.probability_for_word(context[:-1], context[-1])

    def initialize_trie(self):
        logger.info("Create prefix trie")
        extra_chrs = "[]-_/:0123456789'?"
        self.prefix_trie = datrie.BaseTrie(string.ascii_lowercase + extra_chrs)
        for w, id in self.word_dict.items():
            self.prefix_trie[w] = id

    def save_model(self, model_name):
        logger.info("Save model to disk")
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(MODEL_SAVE_PATH+model_name+".json", "w",
                  encoding="latin-1") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(MODEL_SAVE_PATH+model_name+".h5")

        logger.info("Model saved")

    def load_model(self, model_name):
        """Load json and create model.
        """
        logger.info("Load model from disk")
        # This is necessary for the usage in a threaded flask server
        global graph
        global session
        graph = tf.Graph()
        with graph.as_default():
            session = tf.compat.v1.Session()
            with session.as_default():
                # Exclude the extension from the model name and add it separately for
                # each operation
                if ".json" in model_name or ".h5" in model_name:
                    model_name = model_name.replace(".json", "").replace(".h5", "")
                json_file = open(model_name+".json", 'r', encoding="latin-1")
                loaded_model_json = json_file.read()
                json_file.close()
                loaded_model = model_from_json(loaded_model_json)
                # load weights into new model
                loaded_model.load_weights(model_name+".h5")
                self.model = loaded_model
                self.max_sequence_len = self.model.layers[0].input_length + 1

        logger.info("Model loaded")

    def create_plots(self, model_name):
        logger.info("Create plots")
        matplotlib.use('Agg')
        # Accuracy plot
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(INFO_PATH+model_name+'_acc.pdf')
        plt.close()
        # Loss plot
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(INFO_PATH+model_name+'_loss.pdf')

    def write_info(self, model_name):
        logger.info("Write model information to file")
        if self.model and self.history:
            with open(INFO_PATH+model_name+"_info.txt", "w",
                      encoding="latin-1") as file:
                file.write("*"*80+"\n")
                datetime = strftime("%Y-%m-%d %H:%M", localtime())
                file.write("%s %s\n" % (model_name, datetime))
                file.write("*"*80+"\n")
                file.write("\n")

                heading = "Input file:"
                file.write("%s\n" % heading)
                file.write("-"*len(heading)+"\n")
                file.write("Input file name:\t%s\n" % self.input_file)
                file.write("#distinct_words:\t%d\n" % len(self.word_dict))
                file.write("MAX_SEQUENCE_LEN:\t%d\n" % MAX_SEQUENCE_LEN)
                file.write("\n")

                heading = "Training parameters:"
                file.write("%s\n" % heading)
                file.write("-"*len(heading)+"\n")
                if 'batch_size' in self.history.params:
                    batch_size = self.history.params['batch_size']
                else:
                    batch_size = self.batch_size
                file.write("Batch size:\t%d\n" % batch_size)
                file.write("#epochs:\t%d\n" % self.history.params['epochs'])
                if 'samples' in self.history.params:
                    samples = self.history.params['samples']
                else:
                    samples = self.NUM_TRAIN_SAMPLES + self.NUM_VAL_SAMPLES
                file.write("#samples:\t%d\n" % samples)
                file.write("\n")

                heading = "Results:"
                file.write("%s\n" % heading)
                file.write("-"*len(heading)+"\n")
                file.write("Final val_loss:\t%f\n" %
                           self.history.history['val_loss'][-1])
                file.write("Final val_acc:\t%f\n" %
                           self.history.history['val_acc'][-1])
                file.write("\n")

                heading = "Model architecture:\n"
                file.write("%s" % heading)
                self.model.summary(print_fn=lambda x: file.write(x+"\n"))
        else:
            logger.warning("Model or history does not exist.")


def print_usage_and_exit():
    usage_str = ("Usage: python3 %s <training_data_path> [-ich] " +
                 "[-s <model_name>][-l <model_name>]" % sys.argv[0])
    logger.warning(usage_str)
    exit(2)


if __name__ == "__main__":
    # Handle command line arguments
    options = "s:l:c:h"
    long_options = ["save_model", "load_model",
                    "continue_training", "help"]
    try:
        opts, args = getopt.gnu_getopt(sys.argv, options, long_options)
    except getopt.GetoptError:
        logger.error("Error while parsing the command line arguments.")
        print_usage_and_exit()

    save_model = ""
    load_model_path = ""
    continue_training = ""
    for opt, opt_args in opts:
        if opt == '-s' or opt == '--save_model':
            save_model = opt_args
        elif opt == '-l' or opt == '--load_model':
            load_model_path = opt_args
        elif opt == '-c' or opt == '--continue_training':
            continue_training = opt_args
        elif opt == '-h' or opt == '--help':
            hstrng = ("Usage: python3 %s <training_data_path> [arguments]\n\n"
                      "Arguments:\n"
                      "-s, --save_model\t\tSave the trained model to the "
                      "specified path\n"
                      "-l, --load_model\t\tLoad an existing model from the"
                      " specified path\n"
                      "-c, --continue_training\t\tContinue training the"
                      " specified checkpoint of a model\n"
                      "-h, --help\t\tShow help options\n" % args[0])
            logger.warning(hstrng)
            sys.exit(2)
        else:
            print_usage_and_exit()

    if len(args) != 2:
        print_usage_and_exit()

    train_data_path = args[1]
    lm = LM(train_data_path)

    if load_model_path:
        # Load an existing model from file
        # Get the model name in case the user entered a path
        model_name = load_model_path.split("/")[-1]
        model_name = os.path.splitext(model_name)[0]
        lm.load_model(model_name)
    elif continue_training:
        # Continue training an existing hdf5 model
        lm.gen_training_data()
        model_path = CHECKPOINT_LOAD_PATH + continue_training
        lm.model = load_model(model_path)
        lm.train_model()
    else:
        # Train a new model
        lm.gen_training_data()
        lm.create_model()
        lm.train_model()

    if save_model:
        # Save the model
        lm.save_model(save_model)
        lm.create_plots(save_model)
        lm.write_info(save_model)
