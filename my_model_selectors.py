import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_model = None
        best_bic_score = None

        f = len(self.X[0])
        N = len(self.X)

        for m in range(self.min_n_components, self.max_n_components+1):
            logL = None
            try:
                hmm_model = GaussianHMM(n_components=m, covariance_type="diag", n_iter=1000,
                            random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = hmm_model.score(self.X, self.lengths)
                p = math.pow(m,2) + 2 * m * f -1
                new_bic_score = -2 * logL + p * np.log(N)
                if best_model is None or new_bic_score > best_bic_score:
                    best_model = hmm_model
                    best_bic_score = new_bic_score
            except Exception:
                pass
        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        best_model = None
        best_dic_score = None

        numWord = len(self.hwords)

        for p in range(self.min_n_components, self.max_n_components):
            try:
                hmm_model = GaussianHMM(n_components=p, covariance_type="diag", n_iter=1000,
                            random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = hmm_model.score(self.X, self.lengths)

                antiLogL = 0.0
                wc = 0

                for word in self.hwords:
                    if word != self.this_word:
                        word_X, word_lengths = self.hwords[word]
                        antiLogL += hmm_model.score(word_X, word_lengths)

                # Normalize
                antiLogL /= float(numWord-1)

                new_dic_score = logL - antiLogL
                if best_model is None or new_dic_score > best_dic_score:
                    best_model = hmm_model
                    best_dic_score = new_dic_score
            except Exception as e:
                pass
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_logL = None

        for i in range(self.min_n_components, self.max_n_components):
            if(len(self.lengths) < 2):
                try:
                    hmm_model = self.base_model(i)
                    logL = hmm_model.score(self.X, self.lengths)
                    if best_model is None or logL > best_logL:
                        best_model = hmm_model
                        best_logL = logL
                except:
                    pass    
            else:
                split_method = KFold(n_splits=min(3, len(self.lengths)))
                try:
                    for cv_train_idx, cv_test_idx in split_method.split(self.lengths):
                        try:
                            train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                            test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                            hmm_model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(train_X, train_lengths)
                            logL = hmm_model.score(test_X, test_lengths)
                            if best_model is None or logL > best_logL:
                                best_model = hmm_model
                                best_logL = logL
                        except Exception:
                            pass
                except Exception:
                    pass
        return best_model