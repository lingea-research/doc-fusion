from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
import networkx
from pathlib import Path
import json
from nltk import word_tokenize, sent_tokenize

from data.loader import iter_sumec, iter_multi
from fusion import process_doc_collection, view_collection
from rouge_raw import RougeRaw
from rouge_metric import PyRouge


def get_parallel_data(run_id, n_points, dataset):
    from_idx = run_id * n_points
    to_idx = from_idx + n_points
    gen = iter_sumec('test') if dataset == 'sumec' else iter_multi('test')
    for idx, datapoint in enumerate(gen):
        if to_idx > idx >= from_idx:
            yield datapoint


# def eval_hypothesis_sumec(gold, system):
def eval_hypothesis_sumec(clusters, system):
    hypothesis = [" ".join([sent for cluster in clusters for sent in cluster])]
    system = [system]
    results = dict()
    rouge = RougeRaw().corpus(hypothesis, system)
    for metric in ["1", "2", "L"]:
        for mtype in 'prf':
            results[metric + "_" + mtype] = getattr(rouge[metric], mtype)
    return results


def eval_hypothesis_multinews(clusters, system):
    hypothesis = [word_tokenize(sent) for clust in clusters for sent in clust]

    # taking just 300 tokens from they hypothesis to make things comparable to results reported in
    # https://arxiv.org/pdf/1906.01749v3.pdf
    cap, i, capped_hypo = 300, 0, []
    for sent in hypothesis:
        new_sent = []
        for word in sent:
            new_sent.append(word)
            if i == cap:
                break
            i += 1
        capped_hypo.append(new_sent)
        if i == cap:
            break

    reference = [word_tokenize(s) for s in sent_tokenize(system)]
    rouge = PyRouge(rouge_n=(1, 2), rouge_l=False, rouge_w=False,
                    rouge_w_weight=1.2, rouge_s=False, rouge_su=True, skip_gap=4)
    scores = rouge.evaluate_tokenized([capped_hypo], [[reference]])
    results = dict()
    for metric in ["1", "2", "su4"]:
        for mtype in 'prf':
            results[metric + "_" + mtype] = scores[f"rouge-{metric}"][mtype]
    return results


def eval_datapoint(datapoint, model, n_cluster_options, n_sent_options, is_multinews):
    plaintext = datapoint.text
    sents, sim_matrix, cluster_hierarchy = process_doc_collection(plaintext, model)

    results = defaultdict(lambda: np.zeros((len(n_cluster_options), len(n_sent_options)), ))
    for row_id, n_clusters in enumerate(n_cluster_options):
        n_clusters = min(n_clusters, len(sents))
        for col_id, n_sents in enumerate(n_sent_options):
            clusters = view_collection(sents, sim_matrix, cluster_hierarchy, n_clusters, n_sents, 1)
            eval_fn = eval_hypothesis_multinews if is_multinews else eval_hypothesis_sumec
            hypothesis_results = eval_fn(clusters, datapoint.summary)
            # if is_multinews:
            #     hypothesis_results = eval_hypothesis_multinews(clusters, datapoint.summary)
            # else:
            #     hypothesis_results = eval_hypothesis_sumec(clusters, datapoint.summary)
            #     # hypothesis = " ".join([sent for cluster in clusters for sent in cluster])
            #     # hypothesis_results = eval_hypothesis_sumec([datapoint.summary], [hypothesis])
            for k, v in hypothesis_results.items():
                results[k][row_id, col_id] = v
    return results


def eval_batch(batch, model, is_multinews):
    failed_ids = []
    cumul_res = defaultdict(list)
    n = 0
    for i, point in enumerate(batch):
        try:
            res = eval_datapoint(point, model, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], is_multinews)
            for k, v in res.items():
                cumul_res[k].append(v)
            n += 1
        except networkx.PowerIterationFailedConvergence:
            failed_ids.append(i)
            continue

    for k, v in cumul_res.items():
        cumul_res[k] = np.array(v).sum(axis=0).tolist()
    return cumul_res, n, failed_ids


def merge_summary_files(path):
    pth = Path(path)
    assert pth.exists()

    cum_scores = defaultdict(lambda: np.zeros((5, 5), dtype=np.float))
    n_processed, n_failed = 0, 0
    for f in pth.glob('*.jsonl'):
        with f.open('rt') as fin:
            cum_scores_ = json.loads(next(fin))
            for k, v in cum_scores_.items():
                cum_scores[k] += np.array(v)
            info = json.loads(next(fin))
            n_processed += info["processed_points"]
            n_failed += len(info["failed_points_ids"])

    for k, v in cum_scores.items():
        cum_scores[k] = v.tolist()
    f = pth / "_FINAL"
    with f.open('wt') as fout:
        fout.write(json.dumps(cum_scores) + "\n")
        fout.write(
            json.dumps({"processed_points": n_processed, "failed_points_ids": n_failed}) + "\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, help="0-based indexing")
    parser.add_argument("--n_datapoints", type=int, help="#datapoints to process per run")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--starttime", type=str)
    parser.add_argument("--merge_summary_files_pth", type=str, default='',
        help="a directory conatining partial results to be merged into the '_FINAL' file")
    params = parser.parse_args()

    if params.merge_summary_files_pth:
        # Merging already computed results:
        merge_summary_files(params.merge_summary_files_pth)
        exit()

    assert params.dataset in ('sumec', 'multinews')  # dataset lengths: 44453, 5621

    batch = get_parallel_data(params.run_id, params.n_datapoints, params.dataset)

    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    res, n, failed_ids = eval_batch(batch, model, is_multinews=params.dataset == 'multinews')
    f = Path(f"results/{params.dataset}/{params.starttime}")
    f.mkdir(parents=True, exist_ok=True)
    f /= f"{params.run_id}.jsonl"
    with f.open('wt') as fout:
        fout.write(json.dumps(res) + "\n")
        fout.write(json.dumps({"processed_points": n, "failed_points_ids": failed_ids}) + "\n")
