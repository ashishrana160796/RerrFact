import argparse
import jsonlines
import torch

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--rationale-selection', type=str, required=True)
parser.add_argument('--model_n', type=str, default='./saved_models/neutral_classifier')
parser.add_argument('--model_s', type=str, default='./saved_models/support_classifier')
parser.add_argument('--output', type=str, required=True)

args = parser.parse_args()

corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
dataset = jsonlines.open(args.dataset)
rationale_selection = jsonlines.open(args.rationale_selection)
output = jsonlines.open(args.output, 'w')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device "{device}"')



def neutral(evidence, claim, tokenizer, model):


    model.eval()

    def encode(claim, rationale):
        encoding = tokenizer(claim, rationale, padding=True, truncation=True, max_length=512, return_tensors="pt")
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        return input_ids, attention_mask

    def predict(model, evidence, claim):

        with torch.no_grad():
            input_ids, attention_mask = encode(claim, evidence)
            logits = model(input_ids.to(device)).logits
            output = logits.argmax(dim=1).tolist()[0]

        return output

    return predict(model, evidence, claim)

def support(evidence, claim, tokenizer, model):

    model.eval()

    def encode(claim, rationale):
        encoding = tokenizer(claim, rationale, padding=True, truncation=True, max_length=512, return_tensors="pt")
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        return input_ids, attention_mask

    def predict(model, evidence, claim):

        with torch.no_grad():
            input_ids, attention_mask = encode(claim, evidence)
            logits = model(input_ids.to(device)).logits
            output = logits.argmax(dim=1).tolist()[0]

        return output

    return predict(model, evidence, claim)



def detect(evidence, claim, tokenizer_n, model_n, tokenizer_e, model_e):

    # first step
    # predicting neutral or not
    first = neutral(evidence, claim, tokenizer_n, model_n)

    if first == 1:
        final = 1
    else:
        #predicting support or not
        second = support(evidence, claim, tokenizer_e, model_e)
        if second == 0:
            final = 0
        elif second == 1:
            final = 2
    return final



if __name__ == '__main__':

    model_n_path = args.model_n
    tokenizer_n = AutoTokenizer.from_pretrained(model_n_path)
    config_n = AutoConfig.from_pretrained(model_n_path)
    model_n = AutoModelForSequenceClassification.from_pretrained(model_n_path, config=config_n).to(device)

    model_s_path = args.model_s

    tokenizer_s = AutoTokenizer.from_pretrained(model_s_path)
    config_s = AutoConfig.from_pretrained(model_s_path)
    model_s = AutoModelForSequenceClassification.from_pretrained(model_s_path, config=config_s).to(device)

    LABELS = ['CONTRADICT', 'NOT_ENOUGH_INFO', 'SUPPORT']


    for data, selection in tqdm(list(zip(dataset, rationale_selection))):
        assert data['id'] == selection['claim_id']

        claim = data['claim']
        results = {}
        n_not_indices=0
        for doc_id, indices in selection['evidence'].items():
            if not indices:
                results[doc_id] = {'label': 'NOT_ENOUGH_INFO', 'confidence': 1}
                n_not_indices+=1
            else:
                evidence = ' '.join([corpus[int(doc_id)]['abstract'][i] for i in indices])
                label_index = detect(evidence, claim, tokenizer_n, model_n, tokenizer_s, model_s)
                results[doc_id] = {'label': LABELS[label_index]}
        output.write({
            'claim_id': data['id'],
            'labels': results
        })