import argparse
import logging
import re
from score_aggregation_methods import *
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def generatePRFResults(PRF: [int], input_path: str, aggregation):
    fin = open(input_path, "r")
    crumb_lines = fin.readlines()
    filename = input_path.rsplit('/', 1)[1].split('_')
    if 'concat' in filename and 'agg' in filename:
        for prf in tqdm(PRF, desc='Processing PRF'):
            final_res = []
            if aggregation == 'borda':
                final_res = getBERTBorda(prf, input_path)
                final_res = sorted(final_res, key=lambda x: (x[0], x[2]), reverse=True)
                writeFiles(final_res, input_path, aggregation, prf)
                continue
            for line in tqdm(crumb_lines, desc='Processing Lines'):
                segments = line.strip().split('\t')
                qid = segments[0]
                scores = [float(k) for k in segments[1:1+prf]]
                docid = segments[len(segments)-2]
                cumulated_score = 0.0
                if aggregation == 'max':
                    cumulated_score = getMax(scores)
                elif aggregation == 'avg':
                    cumulated_score = getAverage(scores)
                final_res.append([qid, docid, cumulated_score])
            final_res = sorted(final_res, key=lambda x: (x[0], x[2]), reverse=True)
            writeFiles(final_res, input_path, aggregation, prf)
    elif 'concat' in filename and 'chunk' in filename:
        final_res = []
        prf = 0
        for line in tqdm(crumb_lines, desc='Processing Lines'):
            [qid, score, docid, _] = line.strip().split('\t')
            final_res.append([qid, docid, score])
        final_res = sorted(final_res, key=lambda x: (x[0], x[2]), reverse=True)
        for each in filename:
            if each.startswith('p'):
                prf = int(re.findall("(\d+)", each)[0])
        writeFiles(final_res, input_path, aggregation, prf)
    elif 'sliding' in filename:
        final_res = []
        prf = 0
        for each in filename:
            if each.startswith('p'):
                prf = int(re.findall("(\d+)", each)[0])
        if aggregation == 'borda':
            final_res = getBERTBorda(prf, input_path)
            final_res = sorted(final_res, key=lambda x: (x[0], x[2]), reverse=True)
            writeFiles(final_res, input_path, aggregation, prf)
            return
        for line in tqdm(crumb_lines, desc='Processing Lines'):
            segments = line.strip().split('\t')
            qid = segments[0]
            scores = segments[1:len(segments) - 2]
            scores = [float(s) for s in scores]
            docid = segments[len(segments) - 2]
            cumulated_score = 0.0
            if aggregation == 'max':
                cumulated_score = getMax(scores)
            elif aggregation == 'avg':
                cumulated_score = getAverage(scores)
            final_res.append([qid, docid, cumulated_score])
        final_res = sorted(final_res, key=lambda x: (x[0], x[2]), reverse=True)
        writeFiles(final_res, input_path, aggregation, prf)


def generateFirstStagePRFResults(input_path: str, aggregation):
    fin = open(input_path, "r")
    crumb_lines = fin.readlines()
    filename = input_path.rsplit('/', 1)[1].split('_')
    prf = 0
    for each in filename:
        if each.startswith('p'):
            prf = int(re.findall("(\d+)", each)[0])
    if aggregation == 'borda':
        final_res = getBM25Borda(input_path)
        final_res = sorted(final_res, key=lambda x: (x[0], x[2]), reverse=True)
        writeFiles(final_res, input_path, aggregation, prf)
        return
    result_dict = {}
    for line in tqdm(crumb_lines, desc='Processing Lines'):
        [qid_subid, docid, _, score] = line.strip().split('\t')
        [qid, _] = qid_subid.rsplit('_', 1)
        if qid not in result_dict.keys():
            result_dict[qid] = {
                docid: [score]
            }
        else:
            if docid not in result_dict[qid]:
                result_dict[qid][docid] = [score]
            else:
                result_dict[qid][docid].append(score)
    qids = result_dict.keys()
    for q in qids:
        docids = result_dict[q].keys()
        for did in docids:
            scores = result_dict[q][did]
            if len(scores) < prf:
                result_dict[q][did] = scores + ([0.0] * (prf - len(scores)))
    for query_id in qids:
        final_res = []
        doc_ids = result_dict[query_id].keys()
        for d in doc_ids:
            scores = result_dict[query_id][d]
            cumulated_score = 0.0
            if aggregation == 'max':
                cumulated_score = getMax(scores)
            elif aggregation == 'avg':
                cumulated_score = getAverage(scores)
            final_res.append([query_id, d, cumulated_score])
        final_res = sorted(final_res, key=lambda x: (x[0], x[2]), reverse=True)
        writeFiles(final_res, input_path, aggregation, prf)


def writeFiles(res, input_path: str, aggregation, prf):
    path_segs = input_path.rsplit('/', 1)
    filename = input_path.rsplit('/', 1)[1].split('_')
    for index, each in enumerate(filename):
        if each.startswith('p'):
            filename[index] = f'prf{prf}'
    new_filename = '_'.join(filename).rsplit('.', 1)
    if 'sliding' in filename or 'agg' in filename:
        new_filename[1] = f'_{aggregation}.res'
    else:
        new_filename[1] = '.res'
    new_filename[0] = new_filename[0].replace('_crumbs', '')
    path_segs[0] = path_segs[0].replace('/crumb_file', '')
    path_segs[1] = ''.join(new_filename)
    new_output = '/'.join(path_segs)
    # fout = open(f'{new_output.rsplit(".", 1)[0]}.txt', "a+")
    ftrecout = open(new_output, "a+")
    current_id = res[0][0]
    rank = 1
    for item in tqdm(res, desc='Write Files'):
        if item[0] == current_id:
            # fout.write(f'{item[0]}\t{item[1]}\t{rank}\t{item[2]}\n')
            if 'sliding' in filename or 'agg' in filename:
                ftrecout.write(f'{item[0]} Q0 {item[1]} {rank} {item[2]} {aggregation}_{prf}\n')
            else:
                ftrecout.write(f'{item[0]} Q0 {item[1]} {rank} {item[2]} PRF_{prf}\n')
            rank += 1
        else:
            rank = 1
            current_id = item[0]
            # fout.write(f'{item[0]}\t{item[1]}\t{rank}\t{item[2]}\n')
            if 'sliding' in filename or 'agg' in filename:
                ftrecout.write(f'{item[0]} Q0 {item[1]} {rank} {item[2]} {aggregation}_{prf}\n')
            else:
                ftrecout.write(f'{item[0]} Q0 {item[1]} {rank} {item[2]} PRF_{prf}\n')
            rank += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-bm25', help='First Stage PRF Flag', action='store_true', default=False)
    parser.add_argument('-bert', help='Second Stage PRF Flag', action='store_true', default=False)
    parser.add_argument('-i', '--input_path',
                        help='The Input Crumb File (TXT) Path (BERT), Input Res (TXT) File (BM25)', type=str,
                        required=True)
    parser.add_argument('-agg', '--aggregation',
                        help='Aggregation Methods, Can Be max, borda, avg', default='avg')
    args = parser.parse_args()

    BERT_PRF = [1, 3, 5, 10, 15, 20]

    if args.bm25:
        generateFirstStagePRFResults(args.input_path, args.aggregation)
    elif args.bert:
        generatePRFResults(BERT_PRF, args.input_path, args.aggregation)
    else:
        print('Missing PRF Stage Flag.')
