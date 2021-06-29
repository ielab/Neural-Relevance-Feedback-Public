from tqdm import tqdm


def getMax(scores: [float]):
    return max(scores)


def getAverage(scores: [float]):
    return sum(scores) / len(scores)


def getBERTBorda(prf: int, input: str):
    final_res = []
    fin = open(input, "r")
    filename = input.rsplit('/', 1)[1].split('_')
    crumb_lines = fin.readlines()
    crumb_collection = {}
    for line in tqdm(crumb_lines, desc='Creating Borda Dict'):
        crumbs = line.strip().split('\t')
        qid = crumbs[0]
        docid = crumbs[len(crumbs) - 2]
        if 'agg' in filename:
            scores = crumbs[1:len(crumbs) - 2][0:prf]
        else:
            scores = crumbs[1:len(crumbs) - 2]
        if qid not in crumb_collection.keys():
            crumb_collection[qid] = {
                docid: scores
            }
        else:
            if docid not in crumb_collection[qid].keys():
                crumb_collection[qid][docid] = scores
    qids = crumb_collection.keys()
    if 'agg' in filename:
        for each_qid in qids:
            docids = crumb_collection[each_qid].keys()
            voters = []
            final_rank = {}
            for i in range(prf):
                temp_voter = []
                for each_docid in docids:
                    each_scores = crumb_collection[each_qid][each_docid]
                    temp_voter.append([each_docid, float(each_scores[i])])
                voters.append(temp_voter)
            sorted_voters = [sorted(v, key=lambda x: x[1], reverse=True) for v in voters]
            for ind, sv in enumerate(sorted_voters):
                n = len(sv)
                for index, pair in enumerate(sv):
                    rd = index + 1
                    sorted_voters[ind][index][1] = (n - rd + 1) / n
            for vot in sorted_voters:
                for kv in vot:
                    if kv[0] not in final_rank.keys():
                        final_rank[kv[0]] = kv[1]
                    else:
                        final_rank[kv[0]] += kv[1]
            final_rank = {k: v for k, v in sorted(final_rank.items(), key=lambda item: item[1], reverse=True)}
            for ke, va in final_rank.items():
                final_res.append([each_qid, ke, va])
    else:
        docid_voter = {}
        for e_qid in qids:
            docids = list(crumb_collection[e_qid].keys())
            if e_qid not in docid_voter:
                docid_voter[e_qid] = {}
            for e_did in docids:
                if e_did not in docid_voter[e_qid].keys():
                    docid_voter[e_qid][e_did] = 0
            for edid in docids:
                docid_voter[e_qid][edid] += 1
        for each_qid in qids:
            docids = list(crumb_collection[each_qid].keys())
            voters = []
            final_rank = {}
            count = 0
            for i in range(docid_voter[each_qid][docids[count]]):
                temp_voter = []
                for each_docid in docids:
                    each_scores = crumb_collection[each_qid][each_docid]
                    temp_voter.append([each_docid, float(each_scores[i])])
                voters.append(temp_voter)
                count += 1
            sorted_voters = [sorted(v, key=lambda x: x[1], reverse=True) for v in voters]
            for ind, sv in enumerate(sorted_voters):
                n = len(sv)
                for index, pair in enumerate(sv):
                    rd = index + 1
                    sorted_voters[ind][index][1] = (n - rd + 1) / n
            for vot in sorted_voters:
                for kv in vot:
                    if kv[0] not in final_rank.keys():
                        final_rank[kv[0]] = kv[1]
                    else:
                        final_rank[kv[0]] += kv[1]
            final_rank = {k: v for k, v in sorted(final_rank.items(), key=lambda item: item[1], reverse=True)}
            for ke, va in final_rank.items():
                final_res.append([each_qid, ke, va])
    return final_res


def getBM25Borda(input: str):
    final_res = []
    fin = open(input, "r")
    crumb_lines = fin.readlines()
    voter_collection = {}
    for line in tqdm(crumb_lines, desc='Creating Borda Dict'):
        [qid_subid, docid, rank, _] = line.strip().split('\t')
        [qid, subid] = qid_subid.rsplit('_', 1)
        if qid not in voter_collection.keys():
            voter_collection[qid] = {
                subid: {
                    docid: rank
                }
            }
        else:
            if subid not in voter_collection[qid].keys():
                voter_collection[qid][subid] = {
                    docid: rank
                }
            else:
                if docid not in voter_collection[qid][subid].keys():
                    voter_collection[qid][subid][docid] = rank
    qids = voter_collection.keys()
    for each_qid in qids:
        voter_ids = voter_collection[each_qid].keys()
        for each_voter_id in voter_ids:
            docids = voter_collection[each_qid][each_voter_id].keys()
            n = len(docids)
            for each_docid in docids:
                rd = int(voter_collection[each_qid][each_voter_id][each_docid])
                voter_collection[each_qid][each_voter_id][each_docid] = (n - rd + 1) / n
    for query_id in tqdm(qids, desc='Processing Each Query'):
        rank_per_qid = []
        seen = []
        voter_ids = voter_collection[query_id].keys()
        for vid in voter_ids:
            docids = voter_collection[query_id][vid].keys()
            for did in docids:
                if did not in seen:
                    seen.append(did)
                    rank_per_qid.append([query_id, did, voter_collection[query_id][vid][did]])
                else:
                    for ind, pair in enumerate(rank_per_qid):
                        if did == pair[1]:
                            rank_per_qid[ind] = [query_id, did, pair[2] + voter_collection[query_id][vid][did]]
                            break
        rank_per_qid = sorted(rank_per_qid, key=lambda x: x[2], reverse=True)
        final_res += rank_per_qid
    return final_res
