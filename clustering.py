
from typing import List
from sklearn.cluster import AgglomerativeClustering, KMeans, HDBSCAN, OPTICS

import hierarchy

def fit_model(embs, method="agglomerativeClustering", encoder=None, n_clusters=None, n_tokens=None, seeds=[]):
    if method == "agglomerativeClustering":
        return fit_ac(embs, n_clusters)
    elif method == "kmeans":
        return fit_kmeans(embs, n_clusters, seeds, encoder)
    elif method == "hdbscan":
        return fit_hdbscan(embs, n_tokens)
    elif method == "optics":
        return fit_optics(embs, n_tokens)

def fit_ac(embs, n_clusters) -> List[List[List[int]]]:
    ac = AgglomerativeClustering(n_clusters=len(embs), compute_full_tree=True, linkage="ward")
    ac.fit(embs)
    return [ hierarchy.format_ac_cluster_hierarchy(ac)[n_clusters - 1] ]

def fit_kmeans(embs, n_clusters, seeds, encoder) -> List[List[List[int]]]:
    init_centroids = encoder.encode(seeds)
    if seeds != [] and seeds != ['']:
        kmeans = KMeans(n_clusters=n_clusters, init=init_centroids, max_iter=1)
    else:
        kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embs)

    return hierarchy.format_kmeans_cluster_hierarchy(n_clusters, kmeans)

def fit_hdbscan(embs, min_cluster_size=2):
    hdbscan = HDBSCAN(min_cluster_size=min_cluster_size)
    hdbscan.fit(embs)
    return hierarchy.format_kmeans_cluster_hierarchy(len(set(hdbscan.labels_)), hdbscan)

def fit_optics(embs, min_samples=2):
    optics = OPTICS(min_samples=min_samples)
    optics.fit(embs)
    return hierarchy.format_kmeans_cluster_hierarchy(len(set(optics.labels_)), optics)
    