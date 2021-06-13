import sys
import os
import csv
import numpy as np

from src.classification.data import AutoQADataset, AutoQAJsonDataset
# from src.classification.callbacks import CustomFlowCallback
from classification import load_args
from src.classification.custom_trainer import CustomTrainer
from src.classification.utils import (
    collate_fn, compute_metrics, dump_test_results, dir_empty_or_nonexistent
)
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification
)


def main(json_data=None):
    sys.argv.append('call_api')
    args = load_args()

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_dir, num_labels=args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_dir, model_max_length=args.max_seq_length)
    test_dataset = AutoQAJsonDataset(args, json_data, tokenizer)

    # Predict unlabelled data
    predictor = CustomTrainer(
        model,
        args=args,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )        
    outputs = predictor.predict(test_dataset=test_dataset)
    output_json = {**outputs._asdict(), 'predicted_labels': np.argmax(outputs.predictions, axis=1)}

    return output_json


def file_to_json(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for n,line in enumerate(reader):
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
            
    json_data = []
    for line in lines:
        json_data.append({
            'guid': line[0],
            'label': line[1],
            'body': line[2],
        })

    return json_data


def test_call():
    json_data = file_to_json('data/go/autoQA/small/test.tsv')        # TODO: this simulates json data being passed in.
    print(main(json_data))

if __name__ == '__main__':
    test_call()