import torch
import faiss
import argparse
import numpy as np
from tqdm import tqdm
from helper import RepBERT, RepBERT_MSMARCO_DATASET, DEVICE
from torch.utils.data import DataLoader
from transformers import BertConfig


def generate_embeddings(args, model):
    index = faiss.IndexFlatIP(768)
    wrapped_index = faiss.IndexIDMap(index)
    dataset = RepBERT_MSMARCO_DATASET(args.collection_file, args.max_doc_length)
    batch_size = args.per_gpu_batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=4, pin_memory=True)
    for batch in tqdm(dataloader, desc="Generate Passage Embedding"):
        model.eval()
        passage_inputs, attention_mask, pids = batch
        with torch.no_grad():
            output = model(passage_inputs.to(DEVICE), attention_mask.to(DEVICE))
            sequence_embeddings = output.detach().cpu().numpy().astype('float32')
            wrapped_index.add_with_ids(sequence_embeddings, np.array(pids).astype('int64'))
    faiss.write_index(index, f'{args.embedding_output}/RepBERT_FlatIP_collection.index')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--embedding_output", type=str, required=True)
    parser.add_argument("--max_doc_length", type=int, default=256)
    parser.add_argument("--per_gpu_batch_size", default=64, type=int)
    args = parser.parse_args()

    args.n_gpu = torch.cuda.device_count()

    config = BertConfig.from_pretrained(args.model_path)
    config.encode_type = "doc"
    model = RepBERT.from_pretrained(args.model_path, config=config)
    model.to(DEVICE)

    generate_embeddings(args, model)
