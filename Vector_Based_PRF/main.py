import torch
import argparse
from timeit import default_timer as timer
from prf_methods import averagePRF, rocchioPRF
from faiss import read_index
import faiss
from tqdm import tqdm
import numpy as np
from helper import ANCE, ANCE_MSMARCO_DATASET, RepBERT, RepBERT_MSMARCO_DATASET, write_results, DEVICE, pid_mapper
from torch.utils.data import DataLoader
from transformers import BertConfig, RobertaConfig, RobertaTokenizer
from searcher import Searcher

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def generate_query_embedding(args, model):
    query_embeddings = dict()
    dataset = RepBERT_MSMARCO_DATASET(args.query_tsv_file, args.max_query_length)
    batch_size = args.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=4, pin_memory=True)
    for batch in tqdm(dataloader, desc="Generate RepBERT Query Embedding"):
        model.eval()
        query_inputs, attention_mask, qids = batch
        with torch.no_grad():
            output = model(query_inputs.to(DEVICE), attention_mask.to(DEVICE))
            sequence_embeddings = output.detach().cpu().numpy().astype('float32')
            for ind, value in enumerate(qids):
                query_embeddings[qids[ind]] = sequence_embeddings[ind]
    return query_embeddings


def generate_ance_query_embedding(args, model):
    query_embeddings = dict()
    dataset = ANCE_MSMARCO_DATASET(args.tokenizer, args.query_tsv_file, args.max_query_length)
    batch_size = args.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=4,
                            pin_memory=True)
    for batch in tqdm(dataloader, desc="Generating ANCE Query Embedding"):
        model.eval()
        query_inputs, attention_mask, qids = batch
        with torch.no_grad():
            inputs = {
                "input_ids": query_inputs.long(),
                "attention_mask": attention_mask.long()}
            output = model.query_emb(**inputs)
            sequence_embeddings = output.detach().cpu().numpy().astype('float32')
            for ind, value in enumerate(qids):
                query_embeddings[qids[ind]] = sequence_embeddings[ind]
    return query_embeddings


def performPRF(args, searcher, query_embeddings):
    if args.prf == 0 or args.iterations == 0:
        qids, scores, doc_ids, doc_embeddings = searcher.search(query_embeddings, args.hit, args.search_with_gpu)
        write_results(qids, doc_ids, scores, args, iteration=0)
    else:
        if args.converge:
            raise NotImplementedError()
        else:
            qids, scores, doc_ids, doc_embeddings = searcher.search(query_embeddings, args.prf, args.search_with_gpu)
            query = query_embeddings
            print('Finished Iteration 0...')
            if args.store_first_iteration:
                write_results(qids, doc_ids, scores, args, iteration=0)
            for i in range(args.iterations):
                if args.prf_method == "avg":
                    qids, scores, doc_ids, doc_embeddings, new_query_embeddings = averagePRF(query, doc_embeddings, searcher, args)
                    query = new_query_embeddings
                    print(f'Finished Iteration {i + 1}...')
                    write_results(qids, doc_ids, scores, args, iteration=i + 1)
                elif args.prf_method == "sum":
                    raise NotImplementedError()
                elif args.prf_method == "rocchio":
                    qids, scores, doc_ids, doc_embeddings, new_query_embeddings = rocchioPRF(query, doc_embeddings, searcher, args)
                    query = new_query_embeddings
                    print(f'Finished Iteration {i + 1}...')
                    write_results(qids, doc_ids, scores, args, iteration=i + 1)
                else:
                    raise NotImplementedError()


def readVectorPRFRerankBaseFile(base_file_path):
    base_res = {}
    for line in open(base_file_path, 'r').readlines():
        [qid, _, docid, _, _, _] = line.strip().split(' ')
        if qid not in base_res.keys():
            base_res[qid] = [docid]
        else:
            base_res[qid].append(docid)
    return base_res


def performVectorPRFRerank(args, query_embeddings, mapper, index, base_res):
    qids = list(query_embeddings.keys())
    final_res = []

    for qid in qids:
        docids = base_res[qid]
        doc_embs = getDocumentEmbeddings(mapper, docids, index)
        prf_doc_embs = doc_embs[:args.prf]
        query_embedding = query_embeddings[qid]
        if args.prf_method == "avg":
            all_embeddings = np.vstack((query_embedding, prf_doc_embs))
            new_query_embedding = np.mean(all_embeddings, axis=0)
            scores = np.multiply([new_query_embedding] * len(doc_embs), np.array(doc_embs)).sum(axis=1)
            for d_idx, d in enumerate(docids):
                final_res.append([qid, d, float(scores[d_idx])])
        elif args.prf_method == "rocchio":
            if args.has_alpha:
                mean_doc_embeddings = [float(v) * float(args.rocchio_beta) for v in np.mean(prf_doc_embs, axis=0)]
                weighted_query_embeddings = [float(q) * float(1 - args.rocchio_beta) for q in query_embedding]
                new_query_embedding = np.sum(np.vstack((weighted_query_embeddings, mean_doc_embeddings)), axis=0)
                scores = np.multiply([new_query_embedding] * len(doc_embs), np.array(doc_embs)).sum(axis=1)
                for d_idx, d in enumerate(docids):
                    final_res.append([qid, d, float(scores[d_idx])])
            else:
                mean_doc_embeddings = [float(v) * float(args.rocchio_beta) for v in np.mean(prf_doc_embs, axis=0)]
                new_query_embedding = np.sum(np.vstack((query_embedding, mean_doc_embeddings)), axis=0)
                scores = np.multiply([new_query_embedding] * len(doc_embs), np.array(doc_embs)).sum(axis=1)
                for d_idx, d in enumerate(docids):
                    final_res.append([qid, d, float(scores[d_idx])])
    if args.has_alpha:
        beta = '_alpha{:.1f}_beta{}'.format(1 - args.rocchio_beta, args.rocchio_beta)
    elif args.prf_method == "rocchio":
        beta = f'_beta{args.rocchio_beta}'
    else:
        beta = ''
    output = open(
        f'{args.result_output.rsplit(".", 1)[0]}_{args.model}_{args.prf_method}{beta}_prf{args.prf}_1.res',
        'a+')
    desc = f'{args.model}-prf{args.prf}-{args.prf_method}-1'
    final_res = sorted(final_res, key=lambda x: (x[0], x[2]), reverse=True)
    current_qid = final_res[0][0]
    rank = 1
    for r in final_res:
        if r[0] != current_qid:
            rank = 1
            output.write(f'{r[0]} Q0 {r[1]} {rank} {r[2]} {desc}\n')
            rank += 1
            current_qid = r[0]
        else:
            output.write(f'{r[0]} Q0 {r[1]} {rank} {r[2]} {desc}\n')
            rank += 1


def getDocumentEmbeddings(mapper, docids, index):
    doc_embeddings = []
    mapped_docids = []
    if mapper is not None:
        for did in docids:
            mapped_docids.append(mapper[did])
    else:
        mapped_docids = docids

    for mapped_id in mapped_docids:
        doc_embedding = index.reconstruct(int(mapped_id))
        doc_embeddings.append(doc_embedding)

    return doc_embeddings


# docid.shape (64, 1000)
# doc_embeddings.shape (64, 1000, 768)
# qids.shape (64)
def getOracleDocEmeddings(args, qids, docids, doc_embeddings):
    qrels_dict = args.qrels_dict
    new_doc_embeddings = []
    for q_index, qid in enumerate(qids):
        relevant_docids = qrels_dict[qid]
        returned_docids = docids[q_index][:args.prf]
        temp_rel_doc_embeddings = []
        for returned_docid_index, returned_docid in enumerate(returned_docids):
            if str(returned_docid) in relevant_docids:
                temp_rel_doc_embeddings.append(doc_embeddings[q_index][returned_docid_index])
            else:
                continue
        new_doc_embeddings.append(temp_rel_doc_embeddings)
    return new_doc_embeddings


def readQueryCollectionFile(collection_path, query_path):
    query_dict = {}
    collection_dict = {}
    query_total = sum(1 for _ in open(query_path))

    for q_line in tqdm(open(query_path), desc="Load Query Dict", total=query_total):
        [qid, query] = q_line.strip().split('\t')
        query_dict[qid] = query

    collection_total = sum(1 for _ in open(collection_path))
    for c_line in tqdm(open(collection_path), desc="Load Collection Dict", total=collection_total):
        [pid, passage] = c_line.strip().split('\t')
        collection_dict[pid] = passage

    return collection_dict, query_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_tsv_file", default="./data/msmarco_passage/query/query_from_train_doc_query_pairs.tsv", type=str)
    parser.add_argument("--result_output", type=str, help="Write Final Result")
    parser.add_argument("--iteration_output", type=str, default='', help="Store Each Iteration Result File")
    parser.add_argument("--prf_method", default="avg", help="avg, rocchio")
    parser.add_argument("--prf", type=int, default=0, help="0: No PRF, Retrieve Hit, > 0: Perform PRF")
    parser.add_argument("--vector_rerank", action="store_true", default=False, help="Run Vector-Based PRF With Rerank")
    parser.add_argument("--vector_rerank_basefile", type=str, default='', help="The base res file for Vector Based PRF Rerank")
    parser.add_argument("--vector_rerank_mapper", default='', type=str, help="The mapper for vector PRF")
    parser.add_argument("--iterations", type=int, default=0, help="Specify How Many Iterations of PRF to Perform, 0: No PRF")
    parser.add_argument("--rocchio_beta", default=0.1, type=float, help="Beta Coefficient for Rocchio Algorithm")
    parser.add_argument("--has_alpha", action='store_true', help="Use alpha or not in rocchio")
    parser.add_argument("--converge", action="store_true", help="Run PRF Until Result Does Not Change")
    parser.add_argument("--store_first_iteration", action="store_true", help="Store First Iteration Or Not")
    parser.add_argument("--search_with_gpu", action="store_true", help="Use GPU to search or not")
    parser.add_argument("--collection_index", default="./data/trec_deep_2019/index/ANCE_FlatIP_collection_mapped_id.index", type=str)
    parser.add_argument("--model", type=str, default="ANCE", help="RepBERT or ANCE")
    parser.add_argument("--model_path", type=str, default="./ANCE_Model")
    parser.add_argument("--mapper_path", type=str, help="The PID Mapper File for WebAP, and TREC CAsT")
    # Config below usually don't need to be changed
    parser.add_argument("--hit", type=int, default=1000, help="Results returned per query")
    parser.add_argument("--gpu_id", default=0, type=int, help="Faiss GPU ID")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_query_length", type=int, default=64)
    args = parser.parse_args()

    options = vars(args)

    print('------------------------------------------------------')
    print('Configurations:')
    for k, v in options.items():
        print(f'{k}: {v}')
    print('------------------------------------------------------')
    print('Loading Index...')
    start = timer()
    index = read_index(args.collection_index)
    if args.search_with_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, args.gpu_id, index)
    else:
        pass
    end = timer()
    print(f'Index Loaded Took: {end-start} Seconds')
    print('------------------------------------------------------')

    args.emb_index = index

    searcher = Searcher(index)

    start_timer = 0
    if args.model == "RepBERT":
        config = BertConfig.from_pretrained(args.model_path)
        config.encode_type = "query"
        model = RepBERT.from_pretrained(args.model_path, config=config)
        model.to(DEVICE)
        start_timer = timer()
        query_embeddings = generate_query_embedding(args, model)
        if args.vector_rerank:
            base_res = readVectorPRFRerankBaseFile(args.vector_rerank_basefile)
            if 'webap' in args.result_output or 'trec_cast' in args.result_output:
                mapper = pid_mapper(args.vector_rerank_mapper)
            else:
                mapper = None
            performVectorPRFRerank(args, query_embeddings, mapper, index, base_res)
        else:
            performPRF(args, searcher, query_embeddings)
    elif args.model == "ANCE":
        config = RobertaConfig.from_pretrained(
            args.model_path,
        )
        tokenizer = RobertaTokenizer.from_pretrained(
            args.model_path,
            do_lower_case=True,
        )
        args.tokenizer = tokenizer
        model = ANCE.from_pretrained(
            args.model_path,
            config=config,
        )
        model.to(DEVICE)
        start_timer = timer()
        query_embeddings = generate_ance_query_embedding(args, model)
        if args.vector_rerank:
            base_res = readVectorPRFRerankBaseFile(args.vector_rerank_basefile)
            if 'webap' in args.result_output or 'trec_cast' in args.result_output:
                mapper = pid_mapper(args.vector_rerank_mapper)
            else:
                mapper = None
            performVectorPRFRerank(args, query_embeddings, mapper, index, base_res)
        else:
            performPRF(args, searcher, query_embeddings)
    else:
        raise NotImplementedError()

    end_timer = timer()
    print(end_timer-start_timer)
    print('Job Finished.')
