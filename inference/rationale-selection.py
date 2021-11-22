import argparse
import torch
import jsonlines
import os

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, default='./data/corpus.jsonl')
parser.add_argument('--dataset', type=str, default='./data/claims_dev.jsonl')
parser.add_argument('--abstract', type=str, default='./prediction/abstract_retrieval_dev.jsonl')
parser.add_argument('--model', type=str, default='./saved_models/rationale_selection_model')
parser.add_argument('--output', type=str, default='./prediction')




args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device "{device}"')


class SciFactRationaleSelectionDataset(Dataset):
    def __init__(self, corpus: str, dataset: str, abstracts):
        self.samples = []
        abstract_retrieval = jsonlines.open(abstracts)
        dataset = jsonlines.open(dataset)
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
        for data, retrieval in tqdm(list(zip(dataset, abstract_retrieval))):
            assert data['id'] == retrieval['id']

            for doc_id in retrieval['retrieved_doc_ids']:
                doc_id = str(doc_id)
                doc = corpus[int(doc_id)]
                #if the doc is correctly retrieved
                if doc_id in list(data['evidence'].keys()):
                    evidence_sentence_idx = {s for es in data['evidence'][doc_id] for s in es['sentences']}
                else:
                    evidence_sentence_idx = {}

                for i, sentence in enumerate(doc['abstract']):
                    self.samples.append({
                        'claim': data['claim'],
                        'sentence': sentence
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def encode(claims: List[str], sentences: List[str]):
    encoded_dict = tokenizer.batch_encode_plus(
        zip(sentences, claims),
        pad_to_max_length=True,
        return_tensors='pt')
    if encoded_dict['input_ids'].size(1) > 512:
        # Too long for the model. Truncate it
        encoded_dict = tokenizer.batch_encode_plus(
            zip(sentences, claims),
            max_length=512,
            truncation_strategy='only_first',
            pad_to_max_length=True,
            return_tensors='pt')
    encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
    return encoded_dict

def predict(model, dataset):
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=args.batch_size_gpu):
            encoded_dict = encode(batch['claim'], batch['sentence'])
            logits = model(**encoded_dict)[0]
            outputs.extend(logits.argmax(dim=1).tolist())
    return outputs

if __name__ == '__main__':


    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()

    abstract_retrieval = jsonlines.open(args.abstract)
    dataset = jsonlines.open(args.dataset)
    corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
    output = jsonlines.open('{}/rationale_selection.jsonl'.format(args.output), 'w')

    with torch.no_grad():
        for data, retrieval in tqdm(list(zip(dataset, abstract_retrieval))):
            assert data['id'] == retrieval['id']
            claim = data['claim']
            evidence = {}
            for doc_id in retrieval['retrieved_doc_ids']:
                doc_id = str(doc_id)
                doc = corpus[int(doc_id)]
                sents = []

                for i, sentence in enumerate(doc['abstract']):
                    encoded_dict = encode([claim], [sentence])
                    logits = model(**encoded_dict)[0]
                    pred = logits.argmax(dim=1).tolist()[0]
                    if pred == 1:
                        sents.append(i)
                evidence[doc_id] = sents

            output.write({
                'claim_id': retrieval['id'],
                'evidence': evidence
            })