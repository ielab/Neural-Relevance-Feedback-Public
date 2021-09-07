import numpy as np
import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def averagePRF(query_embeddings, doc_embeddings, searcher, args):
    new_query_embeddings = {}
    qids = list(query_embeddings.keys())

    for i, qid in enumerate(qids):
        query_embedding = query_embeddings[qid]
        if len(doc_embeddings[i]) == 0:
            new_query_embeddings[qid] = query_embedding
        else:
            top_doc_embeddings = doc_embeddings[i][:args.prf]
            all_embeddings = np.vstack((query_embedding, top_doc_embeddings))
            new_query_embeddings[qid] = np.mean(all_embeddings, axis=0)

    qids, scores, doc_ids, doc_embeddings = searcher.search(new_query_embeddings, args.hit, args.search_with_gpu)
    return qids, scores, doc_ids, doc_embeddings, new_query_embeddings


def rocchioPRF(query_embeddings, doc_embeddings, searcher, args):
    new_query_embeddings = {}
    qids = list(query_embeddings.keys())

    for i, qid in enumerate(qids):
        query_embedding = query_embeddings[qid]
        if len(doc_embeddings[i]) == 0:
            new_query_embeddings[qid] = query_embedding
        else:
            top_doc_embeddings = doc_embeddings[i][:args.prf]
            if args.has_alpha:
                mean_doc_embeddings = [float(v) * float(args.rocchio_beta) for v in np.mean(top_doc_embeddings, axis=0)]
                weighted_query_embeddings = [float(q) * float(1 - args.rocchio_beta) for q in query_embedding]
                new_query_embeddings[qid] = np.sum(np.vstack((weighted_query_embeddings, mean_doc_embeddings)), axis=0)
            else:
                mean_doc_embeddings = [float(v) * float(args.rocchio_beta) for v in np.mean(top_doc_embeddings, axis=0)]
                new_query_embeddings[qid] = np.sum(np.vstack((query_embedding, mean_doc_embeddings)), axis=0)

    qids, scores, doc_ids, doc_embeddings = searcher.search(new_query_embeddings, args.hit, args.search_with_gpu)
    return qids, scores, doc_ids, doc_embeddings, new_query_embeddings
