import torch
import os
from torch import nn
from tqdm import tqdm
import json
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import BertTokenizer, RobertaForSequenceClassification
from transformers.models.bert.modeling_bert import (BertModel, BertPreTrainedModel)

MODEL_TYPE = 'bert-base-uncased'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOKENIZER = BertTokenizer.from_pretrained(MODEL_TYPE)


class ANCE_MSMARCO_DATASET(Dataset):
    def __init__(self, tokenizer, path, max_length):
        self.path = path
        self.max_length = max_length
        self.pids = []
        self.passages = []
        self.tokenizer = tokenizer

        total_size = sum(1 for _ in open(self.path))
        with open(self.path, 'r') as f:
            for line in tqdm(f, total=total_size, desc=f"Load: {os.path.basename(self.path)}"):
                pid, passage = line.strip().split("\t")
                self.pids.append(pid)
                self.passages.append(passage)

    def __getitem__(self, index):
        pid = self.pids[index]
        passage = self.passages[index]
        # If query, first element of input ids is 0, if passage, first element of input ids is 1
        passage_inputs = self.tokenizer(passage, add_special_tokens=True, return_tensors="pt",
                                        padding='max_length',
                                        max_length=self.max_length, truncation=True)
        return passage_inputs["input_ids"][0], passage_inputs["attention_mask"][0], pid

    def __len__(self):
        return len(self.pids)


class RepBERT_MSMARCO_DATASET(Dataset):
    def __init__(self, path, max_length):
        self.path = path
        self.max_length = max_length
        self.pids = []
        self.passages = []

        total_size = sum(1 for _ in open(path))
        with open(path, 'r') as f:
            for line in tqdm(f, total=total_size, desc=f"Load: {os.path.basename(path)}"):
                [pid, passage] = line.strip().split("\t")
                self.pids.append(pid)
                self.passages.append(passage)

    def __getitem__(self, index):
        pid = self.pids[index]
        passage = self.passages[index]
        passage_inputs = TOKENIZER(passage, add_special_tokens=True, return_tensors="pt", padding='max_length',
                                   max_length=self.max_length, truncation=True)
        return passage_inputs["input_ids"][0], passage_inputs["attention_mask"][0], pid

    def __len__(self):
        return len(self.pids)


def _average_sequence_embeddings(sequence_output, valid_mask):
    flags = valid_mask == 1
    lengths = torch.sum(flags, dim=-1)
    lengths = torch.clamp(lengths, 1, None)
    sequence_embeddings = torch.sum(sequence_output * flags[:, :, None], dim=1)
    sequence_embeddings = sequence_embeddings / lengths[:, None]
    return sequence_embeddings


class RepBERT(BertPreTrainedModel):
    def __init__(self, config):
        super(RepBERT, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()

        if config.encode_type == "doc":
            self.token_type_func = torch.ones_like
        else:
            self.token_type_func = torch.zeros_like

    def forward(self, input_ids, valid_mask):
        token_type_ids = self.token_type_func(input_ids)
        sequence_output = self.bert(input_ids, attention_mask=valid_mask, token_type_ids=token_type_ids)[0]
        text_embeddings = _average_sequence_embeddings(sequence_output, valid_mask)
        return text_embeddings


class EmbeddingMixin:
    """
    Mixin for common functions in most embedding models. Each model should define its own bert-like backbone and forward.
    We inherit from RobertaModel to use from_pretrained
    """

    def __init__(self, model_argobj):
        if model_argobj is None:
            self.use_mean = False
        else:
            self.use_mean = model_argobj.use_mean
        print("Using mean:", self.use_mean)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        # assert isinstance(emb_all, tuple)
        if self.use_mean:
            return self.masked_mean(emb_all[0], mask)
        else:
            return emb_all[:, 0, :]

    def query_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

    def body_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")


class NLL(EmbeddingMixin):
    def forward(
            self,
            query_ids,
            attention_mask_q,
            input_ids_a=None,
            attention_mask_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            is_query=True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        logit_matrix = torch.cat([(q_embs * a_embs).sum(-1).unsqueeze(1),
                                  (q_embs * b_embs).sum(-1).unsqueeze(1)], dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        return (loss.mean(),)


class ANCE(NLL, RobertaForSequenceClassification):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """

    def __init__(self, config, model_argobj=None):
        NLL.__init__(self, model_argobj)
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids.to(DEVICE),
                                attention_mask=attention_mask.to(DEVICE)).last_hidden_state
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)


def load_queries(tokenized_queries):
    queries = dict()
    for tokenized_query in tqdm(tokenized_queries, desc="Processing Query"):
        queries[int(tokenized_query['id'])] = tokenized_query['ids']
    return queries


def write_results(qids, doc_ids, scores, args, iteration):
    if args.has_alpha:
        beta = '_alpha{:.1f}_beta{}'.format(1 - args.rocchio_beta, args.rocchio_beta)
    elif args.prf_method == "rocchio":
        beta = f'_beta{args.rocchio_beta}'
    else:
        beta = ''
    if args.prf > 0 and args.iterations > 0 and args.iteration_output != '':
        if args.prf_method == "bert":
            output = open(
                f'{args.iteration_output.rsplit(".", 1)[0]}_{args.model}_{args.prf_method}_threshold{args.threshold}_prf{args.prf}_{iteration}.res',
                'a+')
            desc = f'{args.model}-prf{args.prf}-{args.prf_method}-threshold{args.threshold}-{iteration}'
        else:
            output = open(
                f'{args.iteration_output.rsplit(".", 1)[0]}_{args.model}_{args.prf_method}{beta}_prf{args.prf}_{iteration}.res',
                'a+')
            desc = f'{args.model}-prf{args.prf}-{args.prf_method}-{iteration}'
    else:
        output = open(args.result_output, 'a+')
        desc = f'{args.model}-FlatIP-Index'
    result = []
    for outter_index, docid_set in enumerate(doc_ids):
        for inner_index, did in enumerate(docid_set):
            result.append([qids[outter_index], did, float(scores[outter_index][inner_index])])
    result = sorted(result, key=lambda x: (x[0], x[2]), reverse=True)
    if 'wikipassageqa' in args.result_output or 'webap' in args.result_output or 'trec_cast' in args.result_output:
        mapper = pid_mapper(args.mapper_path)
    else:
        mapper = None
    current_qid = result[0][0]
    rank = 1
    for res in result:
        if res[0] != current_qid:
            rank = 1
            if mapper is not None:
                output.write(f'{res[0]} Q0 {mapper[int(res[1])]} {rank} {res[2]} {desc}\n')
            else:
                output.write(f'{res[0]} Q0 {res[1]} {rank} {res[2]} {desc}\n')
            rank += 1
            current_qid = res[0]
        else:
            if mapper is not None:
                output.write(f'{res[0]} Q0 {mapper[int(res[1])]} {rank} {res[2]} {desc}\n')
            else:
                output.write(f'{res[0]} Q0 {res[1]} {rank} {res[2]} {desc}\n')
            rank += 1


def pid_mapper(mapper_path):
    file = open(mapper_path, 'r')
    mapper_dict = json.loads(file.read(), object_hook=lambda d: {int(k) if k.isdigit() else k: v for k, v in d.items()})
    return mapper_dict
