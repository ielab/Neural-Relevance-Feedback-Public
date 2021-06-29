import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# export CUDA_VISIBLE_DEVICES=0 on GPU server
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class BERTReranker:
    def __init__(self, CONF):
        self.model = AutoModelForSequenceClassification.from_pretrained(CONF['BERT_MODEL'])
        self.tokenizer = AutoTokenizer.from_pretrained(CONF['BERT_TOKENIZER'])
        self.model.eval()
        self.model.to(DEVICE)
        self.softmax = torch.nn.Softmax(dim=1).to(DEVICE)
        self.max_seq_len = 512

    def batch_inference(self, query, passages):
        batch_size = 32
        queries = [query] * len(passages)
        query_chunks = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
        passage_chunks = [passages[i:i + batch_size] for i in range(0, len(passages), batch_size)]
        final_scores = []
        for index in range(len(query_chunks)):
            inputs = self.tokenizer(query_chunks[index], passage_chunks[index], add_special_tokens=True, return_token_type_ids=True,
                                    max_length=self.max_seq_len, truncation=True, padding=True, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                logits = self.model(inputs["input_ids"],
                                    attention_mask=inputs["attention_mask"],
                                    token_type_ids=inputs["token_type_ids"])[0]
            scores = self.softmax(logits)
            final_scores += scores.detach().cpu().numpy()[:, 1].tolist()
        return final_scores
