import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from copy import deepcopy

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class T5Reranker:
    def __init__(self, CONF, seed: int):
        self.model = T5ForConditionalGeneration.from_pretrained(CONF["T5_MODEL"]).eval().to(DEVICE)
        self.tokenizer = T5Tokenizer.from_pretrained(CONF["T5_TOKENIZER"])
        self.max_seq_len = 512
        set_seed(seed)

    def inference(self, query: str, passages: [str]):
        passages = deepcopy(passages)
        pattern = 'Query: {q} Document: {d} Relevant: </s>'
        final_scores = []
        for text in passages:
            encoding = self.tokenizer.encode_plus(pattern.format(q=query, d=text), pad_to_max_length=True,
                                                  return_tensors='pt')
            input_ids, attention_masks = encoding["input_ids"].to(DEVICE), encoding["attention_mask"].to(DEVICE)
            decode_ids = torch.full((input_ids.size(0), 1),
                                    self.model.config.decoder_start_token_id,
                                    dtype=torch.long).to(input_ids.device)
            outputs = self.model(decoder_input_ids=decode_ids, input_ids=input_ids, attention_mask=attention_masks,
                                 use_cache=True)
            next_token_logits = outputs[0][:, -1, :]
            scores = torch.nn.functional.log_softmax(next_token_logits, dim=1)
            # print(scores[0][1176])  # The score of 'true' token
            # print(scores[0][6136])  # The score of 'false' token
            final_scores.append(scores[0][1176])
        return final_scores

    def batch_inference(self, query: str, passages: [str]):
        batch_size = 32
        passages = deepcopy(passages)
        pattern = 'Query: {q} Document: {d} Relevant: </s>'
        formatted_texts = [pattern.format(q=query, d=passage) for passage in passages]
        formatted_text_chunks = [formatted_texts[i:i + batch_size] for i in range(0, len(formatted_texts), batch_size)]
        final_scores = []
        for index in range(len(formatted_text_chunks)):
            encoding = self.tokenizer.batch_encode_plus(formatted_text_chunks[index], pad_to_max_length=True, truncation=True,
                                                        return_tensors='pt')
            input_ids, attention_masks = encoding["input_ids"].to(DEVICE), encoding["attention_mask"].to(DEVICE)
            decode_ids = torch.full((input_ids.size(0), 1),
                                    self.model.config.decoder_start_token_id,
                                    dtype=torch.long).to(input_ids.device)
            outputs = self.model(decoder_input_ids=decode_ids, input_ids=input_ids, attention_mask=attention_masks,
                                 use_cache=True)
            next_token_logits = outputs[0]
            predictions = [next_token_logits[:, :, 1176], next_token_logits[:, :, 6136]]
            stacked = torch.stack(predictions, 1).squeeze(-1)
            scores = torch.nn.functional.softmax(stacked, dim=1)
            final_scores += scores[:, 0].detach().cpu().numpy()[:, 1].tolist()
        return final_scores
