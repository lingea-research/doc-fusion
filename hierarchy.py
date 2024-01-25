
from typing import List, Dict
from sklearn.cluster import AgglomerativeClustering, KMeans

def format_ac_cluster_hierarchy(fitted_clusters: AgglomerativeClustering
                             ) -> List[List[List[int]]]:
    """
    Transform a fitted instance of AgglomerativeClustering to a list representation of clustering
    hierarchy: each element of the list represents one level in the hierarchy.

    E.g.:
    [
        [[token1, token2, ... tokenN]],                       # all tokens in a single cluster
        [[token1, token3], [token2, token4, ... tokenN]],     # tokens split into two clusters
        ...,
        [[token1], [token2], ..., [tokenN]],                  # each token in its own cluster
    ]

    :param fitted_clusters: a fitted AglomerativeClustering instance
    :return: cluster hierarchy
    """
    n_tokens = len(fitted_clusters.labels_)
    levels = [{i: [i] for i in range(n_tokens)}]
    for i, (a, b) in enumerate(fitted_clusters.children_):
        # copy the last level excluding the clusters to be merged:
        leveldict = {k: v for k, v in levels[-1].items() if k not in (a, b)}
        # merge two clusters and add to the current level:
        leveldict[n_tokens + i] = levels[-1][a] + levels[-1][b]
        levels.append(leveldict)

    # cluster ids are no longer needed, keep only their contents, dict -> list:
    levels = [list(level.values()) for level in levels]

    return list(reversed(levels))

# transform fitted_clusters (at i-th position is the clusted index of the i-th token) to the hierarchy
def format_kmeans_cluster_hierarchy(n_clusters, fitted_clusters) -> List[List[List[int]]]:
    cluster_hierarchy = [[]]
    for _ in range(n_clusters):
        cluster_hierarchy[0].append(list())

    for i in range(len(fitted_clusters.labels_)):
        label = fitted_clusters.labels_[i]
        cluster_hierarchy[0][label].append(i)

    return cluster_hierarchy

def sort_cluster_hierarchy(cluster_hierarchy: List[List[List[int]]],
                           textrank_scores: Dict[int, float]) -> List[List[List[int]]]:
    """
    For each level in the cluster hierarchy, use textrank scores to sort
    1) each unit/token within a cluster
    2) each cluster within the level (by the first element/maximum score)

    :param cluster_hierarchy:
    :param textrank_scores:
    :return:
    """
    sorted_cluster_hierarchy = []
    for level in cluster_hierarchy:
        # sort tokens within each cluster:
        sorted_level_clusters = [sorted(cluster, key=lambda sent: -textrank_scores[sent])
                                 for cluster in level]
        # sort clusters within each level
        sorted_level = sorted(sorted_level_clusters,
                              key=lambda cluster: -textrank_scores[cluster[0]])
        sorted_cluster_hierarchy.append(sorted_level)
    return sorted_cluster_hierarchy