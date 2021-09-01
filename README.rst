deepRL.py
========
This repo contains an implementation of a deep entity resolution algorithm inspired by the paper `Low-resource Deep Entity Resolution with Transfer and Active Learning <https://arxiv.org/pdf/1906.08042.pdf>`__ along with a demonstration in a .ipynb notebook. It deviates from the paper in several ways. This is because the code was written during a research assistant position at The Brown Policy Lab so the code is adapted for our specific use case. The main differences are as follows.

-The fastText model is trained on the datasets provided when the class is initialized and the size of the fastText vectors are customizable. 

-The deepRL class allows the user to build the network with either:

1.  one single BiGRU, both datasets to be matched are passed into the same BiGRU and the absolute difference of the output vecgtors for each column are passed to the MLP layer
2.  two seperate BiGRUs, one for each dataset to be matched. The absolute difference of the output vectors for each column are passed to the MLP layer

Dependencies
============
add dependencies

High Level Summary of Model
============

The paper linked above provides lots of detail about the algorithm but the basic concept is as follows. One of the main issues that arises when working on entity resolution is the fact that generally speaking in real world scenarios, limit labeled data is avialable to train a model. To handle this issue the author proposes a transfer learning approach where the network is initially trained on a 'source' dataset with abundant match/non-match labels. These weights and biases are then transfered over to be used with the 'target' dataset where the labels are not known. In order for this to work the paper proposes using a gradient reversal layer during training to acheive dataset-independent internal representations. If the classifier is performing well on the source dataset and the internal representations are indistiguishable between source and target dataset than we have reason to believe the model will perform reasonably on the target dataset. We predict on the target dataset and pick the highest confidence match and non-match pairs (P(match) closest to 1 and 0), we automatically label them and add them to the training set. We also pull out the lowest confidence pairs (P(match) closest to 0.5) and we labal these by hand. At each iteration we continue to fine tune the model and continue to increase the size of the labeled target dataset. The steps to use the deepRL class are as follows. 

How Does It Work?
=============
There are several different workflows that could be implemented using deepRL. Here we will outline how each method words and the user can customize their process for the specific task. The first step after preprocessing is to initialize the deepRL class. The class takes the following parameters: 

  Parameters
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
    	indices of labeled pairs- indices correspond with candidate_pairs_target
      
Work in Progress!
