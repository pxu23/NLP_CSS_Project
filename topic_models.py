from gensim.models import LdaMulticore
import pandas
from collections import Counter

from tabulate import tabulate
from gensim.corpora import Dictionary
from helper import read_data
from log_odds import compute_odds_with_prior, smoother
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from tqdm import tqdm
tqdm.pandas()

def load_model():
    return LdaMulticore.load("model/lda_model.best")


def load_dictionary():
    return Dictionary.load("model/lda_model.dictionary")


def assign_topics(model, dictionary, data):
    """ Assign topics to documents based on highest scoring topic

    Parameters:
    ===========
    model      (gensim.models.LdaMulticore): trained LDA model
    dictionary (gensim.corpora.Dictionary): dictionary used to train the model
                                            (use Dictionary.doc2bow to convert documents to BOW)
    data        (pandas.DataFrame): pandas dataframe containing a "text" column    

    Returns:
    ========
    topics   (List[int]): a list containing the index of the highest scoring topic per document
    scores   (List[float]): a list containing the scores of the highgest scoring topic per document
    """ 
    topics = []
    scores = []
    
    ####### PART A #######
    for idx, row in data.iterrows():
        doc = row['text']
        #print(f"doc is {doc}")
        words = Dictionary.doc2bow(dictionary, document=doc.split(" "))
        #print(f"words are {words}")
        #print(f"model output is {model[words]}")
        topic_scores = model[words]
        sorted_topic_scores = sorted(topic_scores,key=lambda x: x[1], reverse=True)
        #print(f"sorted_topic_scores is {sorted_topic_scores}")

        highest_topic_score_idx, highest_topic_score = sorted_topic_scores[0]
        topics.append(highest_topic_score_idx)
        scores.append(highest_topic_score)
    
    return topics, scores
    

if __name__ == "__main__":
    data_df = read_data()
    lda = load_model()
    dictionary = load_dictionary()


    topics, scores = assign_topics(lda, dictionary, data_df)
    data_df['topic'] = topics
    data_df['score'] = scores
    
    tax_related = data_df[data_df.topic==5]
    
    # Separate data
    
    r_df = tax_related[tax_related.party == "R"]
    d_df = tax_related[tax_related.party == "D"]

    r_counter = Counter()
    r_df.text.apply(lambda t: r_counter.update(t.split()))

    d_counter = Counter()
    d_df.text.apply(lambda t: d_counter.update(t.split()))
    
    prior = d_counter
    prior.update(r_counter)
    
    d_counters = []
    r_counters = []
    
    for i, session_id in enumerate(tax_related.session_id.unique()):
        r_df = tax_related[(tax_related.party == "R")&(tax_related.session_id == session_id)]
        d_df = tax_related[(tax_related.party == "D")&(tax_related.session_id == session_id)]
        r_counter = Counter()
        r_df.text.apply(lambda t: r_counter.update(t.split()))
        r_counters.append(r_counter)

        d_counter = Counter()
        d_df.text.apply(lambda t: d_counter.update(t.split()))
        d_counters.append(d_counter)
    new_d_counters = smoother(A=.7, window=1, count_list=d_counters)
    new_r_counters = smoother(A=.7, window=1, count_list=r_counters)
    
    print()
    print("Changes over time in log odds with prior")
    political_keywords = {w: [] for w in ['abortion', 'justice', 'freedom']}
    for i, (d_counter, r_counter) in enumerate(zip(new_d_counters, new_r_counters)):
        w_to_ratio = compute_odds_with_prior(d_counter, r_counter, prior)
        for w, odds_list in political_keywords.items():
            political_keywords[w].append(w_to_ratio[w])
    table = []
    for w, odds_list in political_keywords.items():
        table.append([w] + odds_list)
    print(tabulate(table, headers=["Word", "112", "113", "114"]))
    
    
    
   
    
    
    