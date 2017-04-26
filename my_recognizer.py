import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for word_id in test_set.get_all_Xlengths():
        word_x, word_lengths = test_set.get_all_Xlengths()[word_id]
        temp_probabilities = {}
        temp_best_logL = None
        temp_guess_word = None
        for key in models:
            try:
                temp_probabilities[key] = models[key].score(word_x, word_lengths)
                if temp_best_logL is None or temp_probabilities[key] > temp_best_logL:
                    temp_best_logL = temp_probabilities[key]
                    temp_guess_word = key
            except Exception:
                temp_probabilities[key] = None
        probabilities += [temp_probabilities]
        guesses += [temp_guess_word]
    return (probabilities, guesses)

