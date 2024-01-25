
import torch

def similarity_matrix(embs: torch.FloatTensor) -> torch.FloatTensor:
    """
    :param embs: embedded tokens [n tokens x embedding dimensionality]
    :return: cosine similarity matrix [n_tokens x n_tokens]
    """
    norms = torch.linalg.norm(embs, dim=1, ord=2, keepdim=True)
    return (embs @ embs.T) / (norms @ norms.T)

