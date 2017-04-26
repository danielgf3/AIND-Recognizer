import numpy as np
import pandas as pd
from asl_data import AslDb


asl = AslDb() # initializes the database


asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']


from asl_utils import test_features_tryit
# TODO add df columns for 'grnd-rx', 'grnd-ly', 'grnd-lx' representing differences between hand and nose locations
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']


# collect the features into a list
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
 #show a single set of features for a given (video, frame) tuple
[asl.df.ix[98,1][v] for v in features_ground]

training = asl.build_training(features_ground)

df_means = asl.df.groupby('speaker').mean()

asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])

from asl_utils import test_std_tryit
# TODO Create a dataframe named `df_std` with standard deviations grouped by speaker
df_std = asl.df.groupby('speaker').std()



# TODO add features for normalized by speaker values of left, right, x, y
# Name these 'norm-rx', 'norm-ry', 'norm-lx', and 'norm-ly'
# using Z-score scaling (X-Xmean)/Xstd

asl.df['left-y-mean']= asl.df['speaker'].map(df_means['left-y'])
asl.df['right-x-mean']= asl.df['speaker'].map(df_means['right-x'])
asl.df['right-y-mean']= asl.df['speaker'].map(df_means['right-y'])

asl.df['left-x-std']= asl.df['speaker'].map(df_std['left-x'])
asl.df['left-y-std']= asl.df['speaker'].map(df_std['left-y'])
asl.df['right-x-std']= asl.df['speaker'].map(df_std['right-x'])
asl.df['right-y-std']= asl.df['speaker'].map(df_std['right-y'])



asl.df['norm-rx'] = (asl.df['right-x'] - asl.df['right-x-mean']) / asl.df['right-x-std']
asl.df['norm-ry'] = (asl.df['right-y'] - asl.df['right-y-mean']) / asl.df['right-y-std']
asl.df['norm-lx'] = (asl.df['left-x'] - asl.df['left-x-mean']) / asl.df['left-x-std']
asl.df['norm-ly'] = (asl.df['left-y'] - asl.df['left-y-mean']) / asl.df['left-y-std']

features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']



# TODO add features for polar coordinate values where the nose is the origin
# Name these 'polar-rr', 'polar-rtheta', 'polar-lr', and 'polar-ltheta'
# Note that 'polar-rr' and 'polar-rtheta' refer to the radius and angle


asl.df['polar-rr'] = np.sqrt(np.power(asl.df['right-x'] - asl.df['nose-x'], 2) + np.power(asl.df['right-y']- asl.df['nose-y'], 2))
asl.df['polar-rtheta'] = np.arctan2(asl.df['right-x'] - asl.df['nose-x'],asl.df['right-y'] - asl.df['nose-y'])
asl.df['polar-lr'] = np.sqrt(np.power(asl.df['left-x']- asl.df['nose-x'], 2) + np.power(asl.df['left-y']- asl.df['nose-y'], 2))
asl.df['polar-ltheta'] = np.arctan2(asl.df['left-x'] - asl.df['nose-x'],asl.df['left-y'] - asl.df['nose-y'])

features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']


# TODO add features for left, right, x, y differences by one time step, i.e. the "delta" values discussed in the lecture
# Name these 'delta-rx', 'delta-ry', 'delta-lx', and 'delta-ly'

asl.df['delta-rx'] = pd.DataFrame(asl.df['right-x']).diff().fillna(method='backfill')
asl.df['delta-ry'] = pd.DataFrame(asl.df['right-y']).diff().fillna(method='backfill')
asl.df['delta-lx'] = pd.DataFrame(asl.df['left-x']).diff().fillna(method='backfill')
asl.df['delta-ly'] = pd.DataFrame(asl.df['left-y']).diff().fillna(method='backfill')

features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']



# TODO add features of your own design, which may be a combination of the above or something else
# Name these whatever you would like
asl.df.head()
# Feature Scaling
df_groupBySpeaker = asl.df.groupby('speaker')
df_min = df_groupBySpeaker.min()
df_max = df_groupBySpeaker.max()

asl.df['right-x-min']= asl.df['speaker'].map(df_min['right-x'])
asl.df['right-y-min']= asl.df['speaker'].map(df_min['right-y'])
asl.df['left-x-min']= asl.df['speaker'].map(df_min['left-x'])
asl.df['left-y-min']= asl.df['speaker'].map(df_min['left-y'])

asl.df['right-x-max']= asl.df['speaker'].map(df_max['right-x'])
asl.df['right-y-max']= asl.df['speaker'].map(df_max['right-y'])
asl.df['left-x-max']= asl.df['speaker'].map(df_max['left-x'])
asl.df['left-y-max']= asl.df['speaker'].map(df_max['left-y'])

asl.df['scaling-rx'] = (asl.df['right-x'] - asl.df['right-x-min']) / (asl.df['right-x-max'] - asl.df['right-x-min'])
asl.df['scaling-ry'] = (asl.df['right-y'] - asl.df['right-y-min']) / (asl.df['right-y-max'] - asl.df['right-y-min'])
asl.df['scaling-lx'] = (asl.df['left-x'] - asl.df['left-x-min']) / (asl.df['left-x-max'] - asl.df['left-x-min'])
asl.df['scaling-ly'] = (asl.df['left-y'] - asl.df['left-y-min']) / (asl.df['left-y-max'] - asl.df['left-y-min'])
# --

# Normalize polar coordinates
df_mean = df_groupBySpeaker.mean()
df_std = df_groupBySpeaker.std()

asl.df['polar-rr-mean']= asl.df['speaker'].map(df_mean['polar-rr'])
asl.df['polar-rtheta-mean']= asl.df['speaker'].map(df_mean['polar-rtheta'])
asl.df['polar-lr-mean']= asl.df['speaker'].map(df_mean['polar-lr'])
asl.df['polar-ltheta-mean']= asl.df['speaker'].map(df_mean['polar-ltheta'])

asl.df['polar-rr-std']= asl.df['speaker'].map(df_std['polar-rr'])
asl.df['polar-rtheta-std']= asl.df['speaker'].map(df_std['polar-rtheta'])
asl.df['polar-lr-std']= asl.df['speaker'].map(df_std['polar-lr'])
asl.df['polar-ltheta-std']= asl.df['speaker'].map(df_std['polar-ltheta'])

asl.df['norm-polar-rr'] = (asl.df['polar-rr'] - asl.df['polar-rr-mean']) / asl.df['polar-rr-std']
asl.df['norm-polar-rtheta'] = (asl.df['polar-rtheta'] - asl.df['polar-rtheta-mean']) / asl.df['polar-rtheta-std']
asl.df['norm-polar-lr'] = (asl.df['polar-lr'] - asl.df['polar-lr-mean']) / asl.df['polar-lr-std']
asl.df['norm-polar-ltheta'] = (asl.df['polar-ltheta'] - asl.df['polar-ltheta-mean']) / asl.df['polar-ltheta-std']

# --

# More Deltas

asl.df['delta-scaling-rx'] = pd.DataFrame(asl.df['scaling-rx']).diff().fillna(method='backfill')
asl.df['delta-scaling-ry'] = pd.DataFrame(asl.df['scaling-ry']).diff().fillna(method='backfill')
asl.df['delta-scaling-lx'] = pd.DataFrame(asl.df['scaling-lx']).diff().fillna(method='backfill')
asl.df['delta-scaling-ly'] = pd.DataFrame(asl.df['scaling-ly']).diff().fillna(method='backfill')

# TODO define a list named 'features_custom' for building the training set
features_custom = ['scaling-rx', 'scaling-ry', 'scaling-lx', 'scaling-ly', 'norm-polar-rr', 'norm-polar-rtheta', 'norm-polar-lr', 'norm-polar-ltheta', 'delta-scaling-rx', 'delta-scaling-ry', 'delta-scaling-lx', 'delta-scaling-ly']


# P2

import warnings
from hmmlearn.hmm import GaussianHMM

def train_a_word(word, num_hidden_states, features):
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    training = asl.build_training(features)  
    X, lengths = training.get_word_Xlengths(word)
    model = GaussianHMM(n_components=num_hidden_states, n_iter=1000).fit(X, lengths)
    logL = model.score(X, lengths)
    return model, logL

demoword = 'BOOK'
model, logL = train_a_word(demoword, 3, features_ground)
print("Number of states trained in model for {} is {}".format(demoword, model.n_components))
print("logL = {}".format(logL))

def show_model_stats(word, model):
    print("Number of states trained in model for {} is {}".format(word, model.n_components))    
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])    
    for i in range(model.n_components):  # for each hidden state
        print("hidden state #{}".format(i))
        print("mean = ", model.means_[i])
        print("variance = ", variance[i])
        print()


my_testword = 'CHOCOLATE'
model, logL = train_a_word(my_testword, 3, features_custom) # Experiment here with different parameters
show_model_stats(my_testword, model)




from my_model_selectors import SelectorConstant

training = asl.build_training(features_custom)  # Experiment here with different feature sets defined in part 1
word = 'VEGETABLE' # Experiment here with different words
model = SelectorConstant(training.get_all_sequences(), training.get_all_Xlengths(), word, n_constant=3).select()
print("Number of states trained in model for {} is {}".format(word, model.n_components))




from sklearn.model_selection import KFold

training = asl.build_training(features_custom) # Experiment here with different feature sets
word = 'BOOK' # Experiment here with different words
word_sequences = training.get_word_sequences(word)
print(len(word_sequences))
split_method = KFold()
for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
    print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))  # view indices of the folds




words_to_train = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']
import timeit



# ---------


# TODO: Implement SelectorCV in my_model_selector.py
from my_model_selectors import SelectorCV

training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()
for word in words_to_train:
    start = timeit.default_timer()
    
    model = SelectorCV(sequences, Xlengths, word, min_n_components=2, max_n_components=15, random_state = 14).select()
    end = timeit.default_timer()-start
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
    else:
        print("Training failed for {}".format(word))