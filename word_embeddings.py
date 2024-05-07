import numpy as np
import pandas as pd
import gensim
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import urllib.request
import shutil
import os
import random
from copy import deepcopy
from helper import read_data, politics_words

def w2v_to_numpy (model):
    """ Convert the word2vec model (the embeddings) into numpy arrays.
    Also create and return the mapping of words to the row numbers.

    Parameters:
    ===========
    model (gensim.Word2Vec): a trained gensim model

    Returns:
    ========
    embeddings (numpy.ndarray): Embeddings of each word
    idx, iidx (tuple): idx is a dictionary mapping word to row number
                        iidx is a dictionary mapping row number to word
    """ 
    model.wv.fill_norms()
    embeddings = deepcopy(model.wv.get_normed_vectors())
    idx = {w:i for i, w in enumerate (model.wv.index_to_key)}
    iidx = {i:w for i, w in enumerate (model.wv.index_to_key)}
    return embeddings, (idx, iidx)


def near_neighbors (embs, query, word2rownum, rownum2word, k=5):
    """ Get the `k` nearest neighbors for a `query`

    Parameters:
    ===========
    embs (numpy.ndarray): The embeddings.
    query (str): Word whose nearest neighbors are being found
    word2rownum (dict): Map word to row number in the embeddings array
    rownum2word (dict): Map rownum from embeddings array to word
    k (int, default=5): The number of nearest neighbors 

    Returns:
    ========
    neighbors (list): list of near neighbors; 
                    size of the list is k and each item is in the form
                    of word and similarity.
    """

    sims = np.dot (embs, embs[word2rownum[query]])
    indices = np.argsort (-sims)
    return [(rownum2word[index], sims[index]) for index in indices[1:k+1]]


def procrustes(A, B):
    """
    Learn the best rotation matrix to align matrix B to A
    https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    """
    # U, _, Vt = np.linalg.svd(B.dot(A.T))
    U, _, Vt = np.linalg.svd(B.T.dot(A))
    return U.dot(Vt)


def intersect_vocab (idx1, idx2):
  """ Intersect the two vocabularies

  Parameters:
  ===========
  idx1 (dict): the mapping for vocabulary in the first group
  idx2 (dict): the mapping for vocabulary in the second group

  Returns:
  ========
  common_idx, common_iidx (tuple): the common mapping for vocabulary in both groups
  """
  common = idx1.keys() & idx2.keys()
  common_vocab = [v for v in common]

  common_idx, common_iidx = {v:i for i,v in enumerate (common_vocab)}, {i:v for i,v in enumerate (common_vocab)}
  return common_vocab, (common_idx, common_iidx)


def align_matrices (mat1, mat2, idx1, idx2):
  """ Align the embedding matrices and their vocabularies.

  Parameters:
  ===========
  mat1 (numpy.ndarray): embedding matrix for first group
  mat2 (numpy.ndarray): embedding matrix for second group

  index1 (dict): the mapping dictionary for first group
  index2 (dict): the mapping dictionary for the second group

  Returns:
  ========
  remapped_mat1 (numpy.ndarray): the aligned matrix for first group
  remapped_mat2 (numpy.ndarray): the aligned matrix for second group
  common_vocab (tuple): the mapping dictionaries for both the matrices
  """  
  common_vocab, (common_idx, common_iidx) = intersect_vocab (idx1, idx2)
  row_nums1 = [idx1[v] for v in common_vocab]
  row_nums2 = [idx2[v] for v in common_vocab]

  remapped_mat1 = mat1[row_nums1, :]
  remapped_mat2 = mat2[row_nums2, :]
  
  ####### PART B #######
  # use procrustes function here
  Q = procrustes(remapped_mat1, remapped_mat2)

  rotated_mat2 = np.dot(remapped_mat2, Q)

  return remapped_mat1, rotated_mat2, (common_idx, common_iidx)



if __name__ == "__main__":
    data_df = read_data()
    
    r_df = data_df[data_df.party == "R"]
    d_df = data_df[data_df.party == "D"]
    
    query = 'taxes'


    ####### PART A #######
    # parameters to use
    window=50
    min_count=10
    seed=42
    workers=16


    r_model = Word2Vec(window=window,min_count=min_count,seed=seed,workers=workers)
    r_model.build_vocab(r_df.text.apply(lambda t: t.split()))
    #print(f"build vocab done")
    #print(r_model.wv.index_to_key())
    #print(f"r_model.epochs is {r_model.epochs}")
    r_model.train(r_df['text'].apply(lambda t: t.split()), total_examples=r_model.corpus_count,epochs=r_model.epochs)
    r_embs, (r_idx, r_iidx) = w2v_to_numpy(r_model)
    #print("taxes" in r_idx)
    print("PART A: #1")
    print("Republican near neighbors")
    for nbor, sim in near_neighbors(r_embs, query, r_idx, r_iidx, k=10):
        print (nbor, sim)
    print()


    d_model = Word2Vec(window=window,min_count=min_count,seed=seed,workers=workers)
    d_model.build_vocab(d_df.text.apply(lambda t: t.split()))
    #print(d_model.wv.index_to_key)
    #print(f"build vocab done")
    d_model.train(d_df['text'].apply(lambda t: t.split()), total_examples=d_model.corpus_count, epochs=d_model.epochs)
    d_embs, (d_idx, d_iidx) = w2v_to_numpy(d_model)

    #val = "taxes" in d_idx
    #print(val)
    print("Democrat near neighbors")
    for nbor, sim in near_neighbors(d_embs, query, d_idx, d_iidx, k=10):
        print (nbor, sim)


    # Complete Part B.
    dem_aligned_embs, rep_aligned_embs, (common_idx, common_iidx) = align_matrices (d_embs, r_embs, d_idx, r_idx)
    
    ####### PART C1 #######
    print()
    print("Political Words Similarities")
    political_words_sims = [(w, dem_aligned_embs[common_idx[w]].dot(rep_aligned_embs[common_idx[w]])) for w in politics_words] 
    for w,score in sorted (political_words_sims, key=lambda x:x[1], reverse=True):
        print (w, score)
    
    ####### PART C2 #######
    print()
    print("Polarization over time")
    for session_id in data_df.session_id.unique():
        r_df = data_df[(data_df.party == "R")&(data_df.session_id == session_id)]
        d_df = data_df[(data_df.party == "D")&(data_df.session_id == session_id)]
        
        r_model = Word2Vec (r_df.text.apply(lambda t: t.split()), window=50, min_count=10, seed=42, workers=16)
        r_embs, (r_idx, r_iidx) = w2v_to_numpy(r_model)
        
        d_model = Word2Vec (d_df.text.apply(lambda t: t.split()), window=50, min_count=10, seed=42, workers=16)
        d_embs, (d_idx, d_iidx) = w2v_to_numpy(d_model)

        dem_aligned_embs, rep_aligned_embs, (common_idx, common_iidx) = align_matrices (d_embs, r_embs, d_idx, r_idx)
        print(f"Session {session_id}=\t{np.mean([dem_aligned_embs[common_idx[w]].dot(rep_aligned_embs[common_idx[w]]) for w in politics_words])}")
        
        