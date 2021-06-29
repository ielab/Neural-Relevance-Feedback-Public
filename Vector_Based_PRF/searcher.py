import numpy as np
import numpy


class Searcher:
    def __init__(self, index):
        self.index = index

    def search(self, query_embeddings, k, use_gpu=False):
        if use_gpu:
            qids = list(query_embeddings.keys())
            embeddings = np.array(list(query_embeddings.values())).astype('float32')
            scores, ids = self.index.search(embeddings, k)
            p_embeddings = []
            for pids in ids:
                emb = []
                for pid in pids:
                    emb.append(self.index.reconstruct(int(pid)))
                p_embeddings.append(emb)
            return qids, scores, ids, p_embeddings
        if type(query_embeddings) is dict:
            qids = list(query_embeddings.keys())
            embeddings = np.array(list(query_embeddings.values())).astype('float32')
            scores, ids, embeddings = self.index.search_and_reconstruct(embeddings, k)
            return qids, scores, ids, embeddings
        elif type(query_embeddings) is list:
            embeddings = np.array(query_embeddings).astype('float32')
            _, ids, embeddings = self.index.search_and_reconstruct(embeddings, k)
            return ids, embeddings
        elif type(query_embeddings) is numpy.ndarray:
            scores, ids, embeddings = self.index.search_and_reconstruct(query_embeddings, k)
            return scores, ids, embeddings
        else:
            raise TypeError()
