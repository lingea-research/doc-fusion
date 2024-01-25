
from typing import List
import numpy as np

def remove_duplicates(clusters: List[List[int]], sim_matrix: np.ndarray, keep_pairs_lt: float = 1.):
    """
    For each cluster, leave only those tokens/units that are `different enough`
    (i.e. their cosine similarity from others is lower than `keep_pairs_lt`)

    :param clusters:
    :param sim_matrix:
    :param keep_pairs_lt:
    :return:
    """
    different_enough = sim_matrix <= keep_pairs_lt

    res = []
    for cluster in clusters:
        published = []
        for sent in cluster:
            if all([different_enough[sent, sent_published] for sent_published in published]):
                published.append(sent)
        res.append(published)

    return res

# function for transforming a cluster hierarchy into a list of clusters of sentences
def prepare_output(sorted_hierarchy, tokens) -> List[List[str]]:
    result = []
    for cluster in sorted_hierarchy[0]:
        clustered_tokens = []
        for token_id in cluster:
            clustered_tokens.append(tokens[token_id])
        result.append(clustered_tokens)
    return result

def view_collection(tokens, sim_matrix, cluster_hierarchy,
                    n_clusters, n_tokens_per_cluster, keep_pairs_lt) -> List[List[str]]:
    clusters = remove_duplicates(cluster_hierarchy[n_clusters-1], sim_matrix, keep_pairs_lt)

    result = []
    for cluster in clusters:
        result.append([tokens[tok_id] for tok_id in cluster[:n_tokens_per_cluster]])
    return result