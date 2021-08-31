import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import GRU
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from keras.layers import Lambda
import keras
from keras.losses import bce
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from gensim.models import FastText
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import pandas as pd


class DeepRL:
  """
  class to implement deep record linkage

  Methods
  -------
  build_source_model()

    builds the model to be trained on the source dataset

  train_source_model()

    trains the source model on the source dataset

  build_adaptation_model()

    builds the model for dataset adaptation

  train_adaptation_model()

    trains the model for dataset adaptation
    trains on two target variables simultaneously
    one for classifying match/non-match
    one for classifying target dataset/source dataset

  build_target_model()

    builds model to be trained on target dataset

  train_target_model()

    trains target model on target dataset when labels are available

  active_self_learning()

    pulls out the highest confidence pairs and automatically adds to labeled target data
    pulls out lowest confidence pairs for clerical review

  clerical_review()

    conduct clerical review on low confidence pairs identified using the active_self_learning() method

  """
  def __init__(self, df_org_source, df_dup_source, y_source, candidate_pairs_source, df_org_target, df_dup_target, candidate_pairs_target, vec_length, y_target=[], y_target_indices=[]):
    """
    Parameters
    ----------
    df_org_source : dataframe
        first of two fully labeled source datasets
    df_dup_source : dataframe
        second of two fully labeled source datasets
    y_source : list 
        labels for the matching status of pairs in candidate_pairs_source
    candidate_pairs_source : pd.MultiIndex
        pd.MultiIndex object of pairs to be compared from df_org_source and df_dup_source
    df_org_target : dataframe
        first of two target dataframes
    df_dup_target : dataframe
        second of two target dataframes
    candidate_pairs_target : pd.MultiIndex
        pd.MultiIndex object of pairs to be compared from the target datsets
    vec_length : int
        dimension of fastText vectors to be trained with target and source data
    y_target : list, optional
        list of labels for target data
    y_target_indices : list, optional
        indices of labeled pairs, indices correspond with candidate_pairs_target
  
    """
    self.df_org_source = df_org_source
    self.df_dup_source = df_dup_source
    self.y_source = y_source
    self.candidate_pairs_source = candidate_pairs_source
    self.df_org_target = df_org_target
    self.df_dup_target = df_dup_target
    self.candidate_pairs_target = candidate_pairs_target
    self.vec_length = vec_length
    self.model = None
    self.model_adaptation = None
    self.model_target = None
    self.y_target = y_target
    self.y_target_indices = y_target_indices
    self.uncertain = None
    
    # text to train embeddings
    text = []

    # handle case when df_org_source == df_dup_source (if source data is deduplication)
    if self.df_org_source.equals(self.df_dup_source):
      df_list = [self.df_org_source, self.df_org_target, self.df_dup_target]
    else:
      df_list = [self.df_org_source, self.df_dup_source, self.df_org_target, self.df_dup_target]

    # add text from source dataset
    for df in df_list:
      for index in df.index:
        for val in df.loc[index]:
          if isinstance(val, float):
            text.append(['na'])
            continue
          text.append(val.lower().split(' '))

    # add blank token to pad fields so dimensions are equal
    text = text + [['blank']]

    # train fastText
    ft_model = FastText(size=vec_length, min_count=1, seed=33)
    ft_model.build_vocab(text)
    ft_model.train(text, total_examples=ft_model.corpus_count, epochs=20)

    print('training embeddings complete')

    # max_field = the maximum number of words in any field
    self.max_field = np.max([len(val) for val in text])

    # create datasets

    # master lists for org and dup
    org_source = []
    dup_source = []

    # iterate over all candidate pairs
    print('processing source data')
    for org, dup in tqdm(self.candidate_pairs_source):
      org_row = []
      dup_row = []

      # to ensure that source.shape == target.shape we adjust the number columns in the source
      # to match target by either removing or repeating columns
      for i in range(self.df_org_target.shape[1]):
        col = self.df_org_source.columns[i % self.df_org_source.shape[1]]
      
        name_a = self.df_org_source.loc[org, col]

        # 'na' if field == np.nan
        if isinstance(name_a, float):
          name_a = 'na'
        name_a = name_a.split(' ')

        # pad with  'blank' to standardize dimensions
        name_a = ['blank']*(self.max_field - len(name_a)) + name_a

        # add to list for each row
        org_row.append(ft_model.wv[name_a])

        name_b = self.df_dup_source.loc[dup, col]

        # 'na' if field == np.nan
        if isinstance(name_b, float):
          name_b = 'na'
        name_b = name_b.split(' ')

        # pad with  'blank' to standardize dimensions
        name_b = ['blank']*(self.max_field - len(name_b)) + name_b
        dup_row.append(ft_model.wv[name_b])

      # convert to np.array and append
      org_source.append(np.array(org_row))
      dup_source.append(np.array(dup_row))

    print('processing source data complete')

    # master lists for org and dup
    org_target = []
    dup_target = []

    # iterate over all candidate pairs
    print('processing target data')
    for org, dup in tqdm(self.candidate_pairs_target):
      org_row = []
      dup_row = []
      for col in self.df_org_target.columns:
      
        name_a = self.df_org_target.loc[org, col]

        # 'na' if field == np.nan
        if isinstance(name_a, float):
          name_a = 'na'
        name_a = name_a.split(' ')

        # pad with  'blank' to standardize demensions
        name_a = ['blank']*(self.max_field - len(name_a)) + name_a

        # add to list for each row
        org_row.append(ft_model.wv[name_a])

        name_b = self.df_dup_target.loc[dup, col]

        # 'na' if field == np.nan
        if isinstance(name_b, float):
          name_b = 'na'
        name_b = name_b.split(' ')

        # pad with  'blank' to standardize demensions
        name_b = ['blank']*(self.max_field - len(name_b)) + name_b
        dup_row.append(ft_model.wv[name_b])

      # convert to np.array and append
      org_target.append(np.array(org_row))
      dup_target.append(np.array(dup_row))

    print('processing target data complete')

    # save all data as class attributes
    self.org_source_embed = np.array(org_source)
    self.dup_source_embed = np.array(dup_source)
    self.org_target_embed = np.array(org_target)
    self.dup_target_embed = np.array(dup_target)

  def _abs_diff(self, X):
      '''Computes absolute difference for all elements of X'''
      s = X[0]
      for i in range(1, len(X)):
          s -= X[i]
      s = K.abs(s)
      return s

  def _distance_measure(self):
    '''creates model with two BiGRUs, one for each field in a pair
    returns absolute difference of outputs of BiGRUs'''
    inp_a = Input(shape=(None,self.vec_length,))
    inp_b = Input(shape=(None,self.vec_length,))

    # units == self.vec_length//2 so concatenating the outputs in both directions results in 
    # a output of len == self.vec_length//2 if self.vec_length is even
    x = Bidirectional(GRU(self.vec_length//2), merge_mode="concat")(inp_a)

    x = Model(inputs=inp_a, outputs=x)

    y = Bidirectional(GRU(self.vec_length//2), merge_mode='concat')(inp_b)

    y = Model(inputs=inp_b, outputs=y)

    combined = Lambda(lambda x: self._abs_diff(x))([x.output, y.output])

    model = Model(inputs=[inp_a, inp_b], outputs=combined)
    return model

  def _distance_measure2(self):
    '''creates model with a BiGRUs, 
    returns absolute difference of outputs of BiGRU'''
    inp_a = Input(shape=(None,self.vec_length,))
    inp_b = Input(shape=(None,self.vec_length,))

    BiGRU = Bidirectional(GRU(150), merge_mode="concat")

    x = BiGRU(inp_a)

    x = Model(inputs=inp_a, outputs=x)

    y = BiGRU(inp_b)

    y = Model(inputs=inp_b, outputs=y)

    combined = Lambda(lambda x: self._abs_diff(x))([x.output, y.output])

    model = Model(inputs=[inp_a, inp_b], outputs=combined)
    return model

    
  def build_source_model(self, universal=False, summary=True):
    """
    builds the model to be trained on source data

    model saved as self.model

    Parameters
    ----------
    summary : bool, optional
        bool to display model summary or not

    Return 
    ------
    None 
    """
    # GRU layer that takes in 2 inputs
    if universal:
      dist_meas = self._distance_measure2()
    else:
      dist_meas = self._distance_measure()

    # initial inputs
    inp_a = Input(shape=(self.df_dup_target.shape[1], self.max_field, self.vec_length,), name='org_input')
    inp_b = Input(shape=(self.df_dup_target.shape[1], self.max_field, self.vec_length,), name='dup_input')

    # transpose and unstack so now shape == (num_columns, batch_size, max_field, vec_length)
    # allows for iteration over all columns for a pairwise comparison
    inp_a_unstack = tf.unstack(tf.transpose(inp_a, (1, 0, 2, 3)))
    inp_b_unstack = tf.unstack(tf.transpose(inp_b, (1, 0, 2, 3)))

    vec_list = []

    # iterate over each column
    for a, b in zip(inp_a_unstack, inp_b_unstack):
      
      # append results of dist_meas to vec_list
      vec_list.append(dist_meas([a, b]))

    # concatenate vectors in vec_list
    #combined = Concatenate(axis=1)(vec_list)
    combined = tf.keras.layers.add(vec_list)

    z = Dense(300, activation="relu", name='match_classifier_1')(combined)

    # final dense layer for classification
    z = Dense(1, activation="sigmoid", name='match_classifier_2')(z)

    self.model = Model(inputs=[inp_a, inp_b], outputs=z) 

    # if summary == True show summary
    if summary:
      display(self.model.summary())

  class Metrics(Callback):
    """
    custom callback to display f1, recall and precision
    """
    def __init__(self, val_data, split):
      self.val_data = val_data
      self.split = split
    def on_train_begin(self, val_data, logs={}):
      self.val_f1s = []
      self.val_recalls = []
      self.val_precisions = []
    
    def on_epoch_end(self, epoch, logs={}):
      preds = self.model.predict(self.val_data[0])
      if len(preds) == 2:
        preds = preds[0]
      val_predict = (np.asarray(preds)).round()
      val_targ = self.val_data[1]

      val_predict = val_predict[val_targ != 10**6]
      val_targ = val_targ[val_targ != 10**6]

      _val_f1 = f1_score(val_targ, val_predict)
      _val_recall = recall_score(val_targ, val_predict)
      _val_precision = precision_score(val_targ, val_predict)
      self.val_f1s.append(_val_f1)
      self.val_recalls.append(_val_recall)
      self.val_precisions.append(_val_precision)
      if self.split:
        print(f' — val_f1: {_val_f1} — val_precision: {_val_precision} — val_recall {_val_recall}')
      if not self.split:
        print(f' — train_f1: {_val_f1} — train_precision: {_val_precision} — train_recall {_val_recall}')

      return
  
  def train_source_model(self, lr, epochs, batch_size):
    """
    trains source model on source data

    model saved as self.model

    Parameters
    ----------
    lr : float
        learning rate for model
    epochs : int
        number of epochs to train for
    batch_size : int
        batch_size for training

    Returns
    -------
    None 

    """
    # split train/test so we can calculate f1, precision, recall with callback
    org_train, org_test, dup_train, dup_test, y_train, y_test = train_test_split(
      self.org_source_embed, self.dup_source_embed, self.y_source, train_size=0.8, random_state=42)
    
    # pass val data to callback
    metrics = self.Metrics(([org_test, dup_test], np.array(y_test)), True)

    # Configure the model and start training
    optimizer = Adam(learning_rate=lr) 
    self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']) 
    self.model.fit(
      x=[org_train, dup_train], y=np.array(y_train),
      epochs=epochs, batch_size=batch_size, callbacks=[metrics])
    
  def build_adaptation_model(self, summary=True):
    """
    builds the model for adaptations to the target dataset

    saves model as self.model_adaptation

    Parameters
    ----------
    summary : bool
        determines whether or not the model summary is displayed
    
    Returns
    -------
    None

    """

    @tf.custom_gradient
    def _grad_reverse(x):
        y = tf.identity(x)
        def _custom_grad(dy):
            return -dy
        return y, _custom_grad
        
    class GradReverse(tf.keras.layers.Layer):
        def __init__(self):
            super().__init__()

        def call(self, x):
            return _grad_reverse(x)

    inp_a_t = Input(shape=(self.df_dup_target.shape[1], self.max_field, self.vec_length,), name='org_input')
    inp_b_t = Input(shape=(self.df_dup_target.shape[1], self.max_field, self.vec_length,), name='dup_input')

    inp_a_t_unstack = tf.unstack(tf.transpose(inp_a_t, (1, 0, 2, 3)))
    inp_b_t_unstack = tf.unstack(tf.transpose(inp_b_t, (1, 0, 2, 3)))

    vec_list = []

    # iterate over each column
    for a, b in zip(inp_a_t_unstack, inp_b_t_unstack):

        vec_list.append(self.model.layers[6]([a, b]))

    combined = tf.keras.layers.add(vec_list)

    z_match = self.model.layers[8](combined)

    z_match = self.model.layers[9](z_match)

    combined_reverse = GradReverse()(combined)

    z_dataset = Dense(self.vec_length, activation="sigmoid", name='dataset_classifier_1')(combined_reverse)

    z_dataset = Dense(1, activation="sigmoid", name='dataset_classifier_2')(z_dataset)

    self.model_adaptation = Model(inputs=[inp_a_t, inp_b_t], outputs=[z_match, z_dataset])

  def _custom_loss_match(self, y_true, y_pred):
    '''custom loss to set y_true == y_pred wherever
    y_true == 10**6 (meaning true match status is unknown)
    this way we ignore the loss where match status is unknown'''
    bool_mask = tf.not_equal(y_true, 10**6)
    indices = tf.where(bool_mask)
    y_true *= tf.cast(bool_mask, dtype=tf.int64)
    y_pred *= tf.cast(bool_mask, dtype=tf.float32)

    return keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)
  
  def train_adaptation_model(self, lr, epochs, batch_size, match_weight, dataset_weight):

    """
    trains the adaptation model - training is done with two target variables
    one for the match/non-match status and the second for source/target dataset

    Parameters
    ----------
    lr : float
        learning rate
    epochs : int
        number of epochs for training
    batch_size : int
        batch size for training
    match_weight : float
        the weight for the match/non-match loss
        can be used to adjust weights if one classifier is "winning" over the other
    dataset_weight : float
        the weight for the source/target dataset loss
        can be used to adjust weights if one classifier is "winning" over the other

    Returns 
    -------
    None

    """

    # find dataset with least number of pairs
    min_dimension = np.min([self.org_source_embed.shape[0], self.org_target_embed.shape[0]])

    # create new dataset combining source and target, maintain even balance between source and target
    org_adaptation_embed = np.concatenate((self.org_source_embed[:min_dimension], self.org_target_embed[:min_dimension], 
                                           np.take(self.org_target_embed, self.y_target_indices, axis=0)), axis=0)
    dup_adaptation_embed = np.concatenate((self.dup_source_embed[:min_dimension], self.dup_source_embed[:min_dimension], 
                                           np.take(self.dup_target_embed, self.y_target_indices, axis=0)), axis=0)


    # create labels for matching classifier, observations from the target dataset at labeled 10**6
    # we remove these from the loss function for match during training
    y_match = np.array(self.y_source[:min_dimension] + [10**6] * min_dimension + self.y_target)

    # create labels for dataset classifier
    y_dataset = np.array([0]*min_dimension + [1]*(min_dimension + len(self.y_target)))

    print('class balance on y_dataset: ', np.mean(y_dataset))
    print('if dataset classifier in not "winning" the dataset accuracy should be close to class balance')

    # split train/test so we can calculate f1, precision, recall with callback
    org_adapt_train, org_adapt_test, dup_adapt_train, dup_adapt_test, y_match_adapt_train, y_match_adapt_test, y_dataset_adapt_train, y_dataset_adapt_test = train_test_split(
      org_adaptation_embed, dup_adaptation_embed, y_match, y_dataset, train_size=0.8, random_state=42)
    
    # pass val data to callback
    metrics = self.Metrics(([org_adapt_test, dup_adapt_test], np.array(y_match_adapt_test)), True)

    optimizer = Adam(learning_rate=lr)
    self.model_adaptation.compile(optimizer=optimizer, loss={'match_classifier_2':self._custom_loss_match, 'dataset_classifier_2': 'binary_crossentropy'}, 
                    metrics=['accuracy'],
                    loss_weights={"match_classifier_2": match_weight, "dataset_classifier_2": dataset_weight})
    self.model_adaptation.fit(x = {'org_input':org_adapt_train, 'dup_input':dup_adapt_train},
        y = {'match_classifier_2':y_match_adapt_train,'dataset_classifier_2':y_dataset_adapt_train}, 
        epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[metrics])
    
  def build_target_model(self, transfer, universal=False, summary=False):

    """
    builds model for training on the target datset

    Parameters
    ----------
    transfer : bool
        if True use the layers from the adaptation model
        if False random initialization for weights and biases
    summary : bool, optional
        determines whether or not model summary is displayed

    Returns
    -------
    None
    """
    
    if transfer:

      inp_a_t = Input(shape=(self.df_dup_target.shape[1], self.max_field, self.vec_length,), name='org_input')
      inp_b_t = Input(shape=(self.df_dup_target.shape[1], self.max_field, self.vec_length,), name='dup_input')

      inp_a_t_unstack = tf.unstack(tf.transpose(inp_a_t, (1, 0, 2, 3)))
      inp_b_t_unstack = tf.unstack(tf.transpose(inp_b_t, (1, 0, 2, 3)))

      vec_list = []

      # iterate over each column
      for a, b in zip(inp_a_t_unstack, inp_b_t_unstack):

          vec_list.append(self.model_adaptation.layers[6]([a, b]))

      combined = tf.keras.layers.add(vec_list)

      z = self.model_adaptation.layers[9](combined) # match/non-match layer
      z = self.model_adaptation.layers[11](z) # match/non-match layer 2

      #z = self.model_adaptation.layers[10](combined) # source/target layer
      #z = self.model_t.adaptation[14](z) # source/target layer 2

      self.model_target = Model(inputs=[inp_a_t, inp_b_t], outputs=z)
    
    else:
      if universal:
        dist_meas = self._distance_measure2()
      else:
        # GRU layer that takes in 2 inputs
        dist_meas = self._distance_measure()

      # initial inputs
      inp_a = Input(shape=(self.df_dup_target.shape[1], self.max_field, self.vec_length,), name='org_input')
      inp_b = Input(shape=(self.df_dup_target.shape[1], self.max_field, self.vec_length,), name='dup_input')

      # transpose and unstack so now shape == (num_columns, batch_size, max_field, vec_length)
      # allows for iteration over all columns for a pairwise comparison
      inp_a_unstack = tf.unstack(tf.transpose(inp_a, (1, 0, 2, 3)))
      inp_b_unstack = tf.unstack(tf.transpose(inp_b, (1, 0, 2, 3)))

      vec_list = []

      # iterate over each column
      for a, b in zip(inp_a_unstack, inp_b_unstack):
        
        # append results of dist_meas to vec_list
        vec_list.append(dist_meas([a, b]))

      # concatenate vectors in vec_list
      #combined = Concatenate(axis=1)(vec_list)
      combined = tf.keras.layers.add(vec_list)

      z = Dense(self.vec_length, activation="relu", name='match_classifier_1')(combined)

      # final dense layer for classification
      z = Dense(1, activation="sigmoid", name='match_classifier_2')(z)

      self.model_target = Model(inputs=[inp_a, inp_b], outputs=z)
  
    # summary model if display == True
    if summary:
      display(self.model_target.summary())

  def _entropy(self, p):
    '''helper function to compute entropy'''
    return -p * np.log(p) - (1-p) * np.log(1-p)


  def train_target_model(self, lr, epochs, batch_size):

    """
    trains the target model

    Parameters 
    ----------
    lr : float
        learning rate
    epochs : int
        number of epochs for training
    batch_size : int
        batch size
    
    Returns 
    -------
    None

    """
    if self.y_target is None and self.y_target_indices is None:

      print('there is no labeled data')
      print('please run active_self_learning method or submit training data')
      print('the target model should only be trained after there is sufficient target training data')
      return 

    # if there is labeled data, train
    if self.y_target is not None and self.y_target_indices is not None:

      # if there is sufficient labeled data split into train/test for f1, precision and recall calculation
      if len(self.y_target) > 4000:
        org_target_train, org_target_test, dup_target_train, dup_target_test, y_train, y_test = train_test_split(
        np.take(self.org_target_embed, self.y_target_indices, axis=0),np.take(self.dup_target_embed, self.y_target_indices, axis=0), 
        self.y_target, train_size=0.8, random_state=42)

        # calculate f1, precision and recall using test data
        metrics = self.Metrics(([org_target_test, dup_target_test], np.array(y_test)), True)

      # if there is limited training data, train on whole data set 
      # report f1, recall, precision using training data
      else:
        print('NOTE: Limited training data is available')
        print('f1, precision and recall are calculated using training data')
        print('after len(labeled data) > 4000 this function will report test f1, precision and recall')
        org_target_train = np.take(self.org_target_embed, self.y_target_indices, axis=0)
        dup_target_train = np.take(self.dup_target_embed, self.y_target_indices, axis=0)
        y_train = self.y_target

        # calculate f1, precision and recall with training data
        metrics = self.Metrics(([org_target_train, dup_target_train], np.array(y_train)), False)
        
      # train on data with labels
      optimizer = Adam(learning_rate=lr)
      self.model_target.compile(loss=
              keras.losses.BinaryCrossentropy(from_logits=False),
              optimizer=optimizer, metrics=['accuracy'])
      self.model_target.fit(x = {
          'org_input':np.take(self.org_target_embed, self.y_target_indices, axis=0), 
          'dup_input':np.take(self.dup_target_embed, self.y_target_indices, axis=0)
          },
          y = np.array(self.y_target),
        epochs=epochs, batch_size=batch_size, callbacks=[metrics])
    
          
  def active_self_learning(self, n_certain_false, n_certain_true, n_uncertain):

    """
    automatically labels high confidence pairs and adds to target training data
    allows user to hand label uncertain pairs using the clerical review method

    Parameters
    ----------
    n_certain_false : int
        the number of high confidence false pairs to automatically label and add to training set
        should be selected to preserve approximate class balance
    n_certain_true : int
        the number of high confidence true pairs to automatically label and add to training set
        should be selected to preserve approximate class balance
    n_uncertain : int
        the number of low confidence pairs to label by hand

    Returns 
    -------
    None
    """
    
    # if labeled data already exists we don't want to predict on these observations 
    # for transfer and active learning        
    # create list of indices that corresponds with y_pred for unlabeled data
    no_label_indices = np.delete(np.array([i for i in range(self.org_target_embed.shape[0])]), self.y_target_indices)
          
    # probs for unlabaled data
    probs = self.model_target.predict([np.delete(self.org_target_embed, self.y_target_indices, axis=0), 
                                      np.delete(self.dup_target_embed, self.y_target_indices, axis=0)])

    # pull out pairs with highest entropy
    ent_sorted = sorted([(self._entropy(prob), ind) for prob, ind in zip(probs, no_label_indices)])
    self.uncertain = [ind for _, ind in ent_sorted[-n_uncertain:]]

    # sort by prob and pull out certain pairs 
    prob_sorted = sorted([(prob, ind) for prob, ind in zip(probs, no_label_indices)])

    #n_certain_false low probability pairs and n_certain_true high prob pairs are pulled out
    certain = [ind for _, ind in prob_sorted[:n_certain_false]] + [ind for _, ind in prob_sorted[-n_certain_true:]]

    # check if certain and uncertain intersect
    # this happens in the case where our most confident positive cases still have very low prob
    # in this situation we don't add the most certain cases - instead we rely on the clerical_review function
    cardinality = len([i for i in self.uncertain if i in set(certain)])
    if (cardinality == 0) and (prob_sorted[-n_certain_true][0] > 0.5):
        # add certain pairs to training data
        self.y_target_indices = self.y_target_indices + certain
        new_labels = [0]*n_certain_false + [1]*n_certain_true
        self.y_target = self.y_target + new_labels

    # if prob of most confident pair is very low instruct user to perform clerical review
    else:
      print('prob of most confident examples was very low!')
      print('no pairs were added to training data')
      print('use clerical_review function')

  def clerical_review(self):
    """
    used to review the n_uncertain low confidence pairs that were selected using the active_self_learning() method

    Returns
    -------
    None

    """

    # bool to keep track of early exit
    early_exit = False

    # keep track of pairs that are labeled in clerical review
    labeled_list = []
    labeled_list_y = []

    # return none if n_uncertain is set to False
    #if not self.n_uncertain:
    #  print('n_uncertain set to False')
    #  return None

    # iterate through rows of df_uncertain
    if not self.uncertain:
      print('no uncertain pairs to label')
      return
    for i in self.uncertain:
      while True:
          # pull out indices of pair and display data from original dfs
          org1_index, org2_index = self.candidate_pairs_target[i]
          display(pd.concat([self.df_org_target.loc[org1_index,:], 
                            self.df_dup_target.loc[org2_index, :]], axis=1).T)
          
          # label the data True/False or quit
          label = input("Enter label (True/False) :")
          if label == 'quit':
            early_exit = True
            break
          elif label == 'True':
            label = 1
          elif label == 'False':
            label = 0

          if (label == 1) or (label == 0):
            # add labeled pair to labeled_list
            self.y_target_indices.append(i)
            self.y_target.append(label)

            break

          else:
            print('invalid entry - try again \n enter "quit" to terminate loop')

      if label == 'quit':
          break

    if early_exit:
      self.uncertain = self.uncertain[self.uncertain.index(i)]
      print('loop terminated early')

    else:
      self.uncertain = None
      print('all data labeled')
