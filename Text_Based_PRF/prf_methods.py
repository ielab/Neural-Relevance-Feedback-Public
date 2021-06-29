from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')


class PRF_METHODS:
    def __init__(self, prf_k, QUERY, RANKED_FILE, COLLECTION_CONTENT):
        self.prf_k = prf_k
        self.QUERY = QUERY
        self.RANKED_FILE = RANKED_FILE
        self.COLLECTION_CONTENT = COLLECTION_CONTENT

    ####
    # MS MARCO Passage: Average Passage Length = 65 (64.65), Stride = 32
    # TREC CAsT: Average Passage Length = 69 (68.64), Stride = 34
    # TREC DEEP 2019: Average Passage Length = 65 (64.65), Stride = 32
    # WebAP: Average Passage Length = 75 (74.5), Stride = 37
    # WikiPassageQA: Average Passage Length = 134 (134.2), Stride = 67
    ####
    def concat_truncate_prf(self):
        json_queries = []
        tsv_queries = []
        query_dict = {}
        for query in tqdm(self.RANKED_FILE, desc='Processing Concat Chunk Query'):
            qid = query[0][0]
            init_query_content = self.QUERY[qid]
            for i in range(self.prf_k):
                if i < len(query):
                    init_query_content = f'{init_query_content} {self.COLLECTION_CONTENT[query[i][1]]}'
            tokens = word_tokenize(init_query_content)
            temp_tsv = f'{qid}\t{" ".join(tokens[:256])}'
            temp_json = {
                'id': qid,
                'contents': " ".join(tokens[:256])
            }
            query_dict[qid] = [init_query_content]
            tsv_queries.append(temp_tsv)
            json_queries.append(temp_json)
        return tsv_queries, json_queries, query_dict

    def concat_aggregate_prf(self):
        json_queries = []
        tsv_queries = []
        query_dict = {}
        for query in tqdm(self.RANKED_FILE, desc='Processing Concat Agg Query'):
            qid = query[0][0]
            if qid not in query_dict.keys():
                query_dict[qid] = []
            init_query_content = self.QUERY[qid]
            for i in range(self.prf_k):
                if i < len(query):
                    query_formation = f'{init_query_content} {self.COLLECTION_CONTENT[query[i][1]]}'
                    tokens = word_tokenize(query_formation)
                    temp_query = f'{qid}_{i + 1}\t{" ".join(tokens[:256])}'
                    temp_json = {
                        'id': f'{qid}_{i + 1}',
                        'contents': f'{" ".join(tokens[:256])}'
                    }
                    tsv_queries.append(temp_query)
                    json_queries.append(temp_json)
                    query_dict[qid].append(f'{" ".join(tokens[:256])}')
                else:
                    continue
        return tsv_queries, json_queries, query_dict

    ####
    # MS MARCO Passage: Average Passage Length = 65 (64.65), Stride = 32
    # TREC CAsT: Average Passage Length = 69 (68.64), Stride = 34
    # TREC DEEP 2019: Average Passage Length = 65 (64.65), Stride = 32
    # WebAP: Average Passage Length = 75 (74.5), Stride = 37
    # WikiPassageQA: Average Passage Length = 134 (134.2), Stride = 67
    ####
    def sliding_window_prf(self, window_size, stride):
        json_queries = []
        tsv_queries = []
        query_dict = {}
        for query in tqdm(self.RANKED_FILE, desc='Processing Sliding Window Query'):
            qid = query[0][0]
            if qid not in query_dict.keys():
                query_dict[qid] = []
            init_query_content = self.QUERY[qid]
            passage_concatenation = ''
            for i in range(self.prf_k):
                if i < len(query):
                    passage_concatenation = f'{passage_concatenation} {self.COLLECTION_CONTENT[query[i][1]]}'
                else:
                    continue
            tokens = word_tokenize(passage_concatenation)
            for index, i in enumerate([k for k in range(0, len(tokens), stride)]):
                segment = ' '.join(tokens[i:i + window_size])
                formatted_query = f'{init_query_content} {segment}'
                per_query_tokens = word_tokenize(formatted_query)
                temp_tsv = f'{qid}_{index + 1}\t{" ".join(per_query_tokens[:256])}'
                temp_json = {
                    'id': f'{qid}_{index + 1}',
                    'contents': f'{" ".join(per_query_tokens[:256])}'
                }
                tsv_queries.append(temp_tsv)
                json_queries.append(temp_json)
                query_dict[qid].append(f'{" ".join(per_query_tokens[:256])}')
                if i + window_size >= len(tokens):
                    break
        return tsv_queries, json_queries, query_dict
