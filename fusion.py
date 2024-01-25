
from typing import List
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize as tokenize

import clustering, hierarchy, textrank, utils

def summarize(plaintext: str, encoder: SentenceTransformer, 
        full_textrank=False, n_clusters=0, n_tokens=2, method="agglomerativeClustering", seeds=[]) -> List[List[str]]:
    """
    Processes an input text and clusters the sentences based on their embedding cosine similarity.
    Sentences inside the clusters are ordered based on the textrank.

    Parameters
    ----------
    plaintext : str
        Text of the document to summarize.
    encoder : SentenceTransformer
        Transformer for computing the sentence embeddings.
    full_textrank : bool, optional
        Whether to compute textrank for each pair of sentences (True), or only for sentences inside the cluster (False) (default is True)
    n_cluster : int, optional
        Amount of clusters to use, by default sqrt(num_of_sentences) is used.
    method : str, optional
        Clustering method to use. (default is agglomerativeClustering)
    seeds : List[str], optional
        Initial seeds for clustering, supported for KMeans method. (default is [])

    Returns
    -------
    List[List[str]]
        A list of clusters. A cluster is a list of sentences (strings).
    """

    # tokenize using nltk - token = sentence
    tokens = tokenize(plaintext)    

    # if n_clusters was not supplied, take sqrt(num_of_sentences) instead
    if n_clusters == 0:
        n_clusters = int(len(tokens) ** 0.5)

    # compute embeddings
    embs = encoder.encode(tokens)   

    # compute token clusters
    cluster_hierarchy = clustering.fit_model(embs, method, encoder=encoder, n_clusters=n_clusters, n_tokens=n_tokens, seeds=seeds) 

    # compute textrank for each token
    textrank_scores = textrank.get_textrank(embs, full_textrank=full_textrank, cluster_hierarchy=cluster_hierarchy) 
    
    # sort tokens inside clusters using textrank score
    sorted_hierarchy = hierarchy.sort_cluster_hierarchy(cluster_hierarchy, textrank_scores) 

    return utils.prepare_output(sorted_hierarchy, tokens)

