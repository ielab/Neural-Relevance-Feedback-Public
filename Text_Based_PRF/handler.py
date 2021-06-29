import numpy as np
from tqdm import tqdm
from prf_methods import PRF_METHODS


def performPRFFromRankedFile(QUERY, CONF, RANKED_FILE, COLLECTION_CONTENT, RERANKER):
    PRF = int(CONF["PRF"])
    prf_method = CONF["PRF_METHOD"]
    prfer = PRF_METHODS(PRF, QUERY, RANKED_FILE, COLLECTION_CONTENT)
    query_dict = {}
    if prf_method == "CT":
        _, _, query_dict = prfer.concat_truncate_prf()
    elif prf_method == "CA":
        _, _, query_dict = prfer.concat_aggregate_prf()
    elif prf_method == "SW":
        _, _, query_dict = prfer.sliding_window_prf(CONF[CONF["FILETYPE"]]["WINDOW_SIZE"], CONF[CONF["FILETYPE"]]["STRIDE"])
    for query in tqdm(RANKED_FILE, desc='Processing Query'):
        qid = query[0][0]
        docids = []
        queries = query_dict[qid]
        docs = []
        for document in query:
            docid = document[1]
            docids.append(docid)
            doc_content = COLLECTION_CONTENT[docid]
            docs.append(doc_content)
        processResultAndWriteFile(qid, queries, docs, docids, RERANKER, CONF)


def performRerank(QUERY, CONF, RANKED_FILE, COLLECTION_CONTENT, RERANKER):
    for query in tqdm(RANKED_FILE, desc='Processing Query'):
        qid = query[0][0]
        init_query_content = QUERY[qid]
        docids = []
        queries = []
        docs = []
        queries.append(init_query_content)
        for document in query:
            docid = document[1]
            docids.append(docid)
            doc_content = COLLECTION_CONTENT[docid]
            docs.append(doc_content)
        all_scores = []
        score_doc_pairs = []
        for each in tqdm(queries, desc=f'{CONF["MODEL"].upper()} Inference'):
            scores = RERANKER.batch_inference(each, docs)
            all_scores.append(scores)
        array = np.array(all_scores)
        final_scores = array.sum(axis=0)
        for index, did in enumerate(docids):
            score_doc_pairs.append([did, final_scores[index]])
        score_doc_pairs = sorted(score_doc_pairs, key=lambda x: x[1], reverse=True)
        fout = open(f'{CONF[CONF["FILETYPE"]]["RESULT"]}/bm25_default_bert.res', 'a+')
        ftxtout = open(f'{CONF[CONF["FILETYPE"]]["RESULT"]}/bm25_default_bert.txt', 'a+')
        for index, score_doc_pair in enumerate(score_doc_pairs):
            fout.write(f'{qid} Q0 {score_doc_pair[0]} {index + 1} {score_doc_pair[1]} bm25_default_bert\n')
            ftxtout.write(f'{qid}\t{score_doc_pair[0]}\t{index + 1}\t{score_doc_pair[1]}\n')
        fout.close()
        ftxtout.close()


def performPRFWithPyserini(QUERY, RETRIEVER, RERANKER, CONF):
    PRF = int(CONF["PRF"])
    qids = QUERY.keys()
    for qid in tqdm(qids, desc='Processing Query'):
        query = QUERY[qid]
        results = RETRIEVER.retrieve(query)
        docids = []
        queries = []
        docs = []
        for i in range(PRF):
            temp_query = f'{query} {results[i].text}'
            queries.append(temp_query)
        for item in results:
            docids.append(item.id)
            docs.append(item.text)
        processResultAndWriteFile(qid, queries, docs, docids, RERANKER, CONF)


def processResultAndWriteFile(qid, queries, docs, docids, RERANKER, CONF):
    all_scores = []
    # score_doc_pairs = []
    for each in tqdm(queries, desc=f'{CONF["MODEL"].upper()} Inference'):
        scores = RERANKER.batch_inference(each, docs)
        all_scores.append(scores)
    # array = np.array(all_scores)
    # final_scores = array.sum(axis=0)
    # for index, did in enumerate(docids):
    #     score_doc_pairs.append([did, final_scores[index]])
    # score_doc_pairs = sorted(score_doc_pairs, key=lambda x: (x[1]), reverse=True)
    writeBreadCrumbScores(all_scores, docids, qid, CONF)
    # with open(f'{CONF[CONF["FILETYPE"]]["RESULT"]}/{CONF["MODEL"]}_prf{CONF["PRF"]}_{CONF["MODE"]}_aggregation_full.res', 'a+') as fout, \
    #         open(f'{CONF[CONF["FILETYPE"]]["RESULT"]}/{CONF["MODEL"]}_prf{CONF["PRF"]}_{CONF["MODE"]}_aggregation_full.trec.res', 'a+') as ftrecout:
    #     for ind, item in tqdm(enumerate(score_doc_pairs), desc='Writing Res File: '):
    #         fout.write(f'{qid}\t{item[0]}\t{ind + 1}\t{item[1]}\n')
    #         ftrecout.write(f'{qid} Q0 {item[0]} {ind + 1} {item[1]} {CONF["MODEL"]}_PRF{CONF["PRF"]}\n')
    # fout.close()


def writeBreadCrumbScores(all_scores, docids, qid, CONF):
    method = ''
    if CONF["PRF_METHOD"] == "CC":
        method = 'concat_chunk'
    elif CONF["PRF_METHOD"] == "CA":
        method = 'concat_agg'
    elif CONF["PRF_METHOD"] == "SW":
        if CONF["FILETYPE"] == "TREC":
            method = 'sliding_window.w65.s32'
        elif CONF["FILETYPE"] == "CAST":
            method = 'sliding_window.w69.s34'
        elif CONF["FILETYPE"] == "WEBAP":
            method = 'sliding_window.w75.s37'
        elif CONF["FILETYPE"] == "WIKIPASSAGEQA":
            method = 'sliding_window.w134.s67'
    if 'rm3' not in CONF[CONF["FILETYPE"]]["RANK_FILE"]:
        filename = f'{CONF[CONF["FILETYPE"]]["RESULT"]}/bm25_default_bert_prf{CONF["PRF"]}_{method}_crumbs.txt'
    else:
        filename = f'{CONF[CONF["FILETYPE"]]["RESULT"]}/bm25_rm3_default_bert_prf{CONF["PRF"]}_{method}_crumbs.txt'
    with open(filename, 'a+') as fout:
        for ind, docid in enumerate(docids):
            scores = []
            for sub_scores in all_scores:
                scores.append(sub_scores[ind])
            scores_str = '\t'.join([str(f) for f in scores])
            line = f'{qid}\t{scores_str}\t{docid}\tTOP{CONF["PRF"]}\n'
            fout.write(line)
    fout.close()
