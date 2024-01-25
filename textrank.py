
import torch
import networkx
from typing import Dict
import numpy as np

import tensor_math

def textrank(sim_matrix: np.ndarray, max_iter: int = 500, tol: float = 1e-02) -> Dict[int, float]:
    """
    Get a textrank score for each token:
    https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html

    :param sim_matrix: n_tokens x n_tokens matrix of cosine similarities
    :param max_iter: max nb. of steps in solver
    :param tol: convergence check
    :return: {token_id: textrank score, ...}
    """
    graph = networkx.from_numpy_array(sim_matrix)
    return networkx.pagerank_numpy(graph) # networkx.pagerank(graph, max_iter=max_iter, tol=tol)

# function for computing the textrank scores for the embedded tokens
#
# if full_textrank option is set to true, textrank will be computed 'globally' - for all tokens at once
# otherwise, textrank will be computed only for the tokens inside their cluster
#
# warning: full_textrank option is extremely memory inefficient
def get_textrank(embs, full_textrank=True, cluster_hierarchy=None):
    if full_textrank:
        sim_matrix = tensor_math.similarity_matrix(torch.FloatTensor(embs)).detach().numpy()
        return textrank(sim_matrix)
    
    textrank_scores = dict()
    for cluster in cluster_hierarchy[0]:
        cluster_embs = []        
        i = 0
        token_dict = dict()
        for token_id in cluster:
            cluster_embs.append(embs[token_id])
            token_dict[i] = token_id
            i += 1
            
        sim_matrix = tensor_math.similarity_matrix(torch.FloatTensor(cluster_embs)).detach().numpy()
        cluster_textrank = textrank(sim_matrix)
        for token_id in cluster_textrank:
            textrank_scores[token_dict[token_id]] = cluster_textrank[token_id]
    
    return textrank_scores
    