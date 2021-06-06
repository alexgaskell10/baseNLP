import sys
import numpy as np
from random import shuffle

from src.classification.data import AutoQADataset
from src.classification.callbacks import CustomFlowCallback
from classification import classification_loop, load_args
from src.classification.custom_trainer import CustomTrainer
from src.classification.utils import (
    collate_fn, compute_metrics, dump_test_results, dir_empty_or_nonexistent
)

def main():
    sys.argv.append('self_training')
    args = load_args()
    # args.do_train = False   # TODO
    # args.do_eval = False   # TODO
    # args.do_predict = False   # TODO
    teacher, tokenizer, datasets = classification_loop(args)

    print('\n'*3, 'Teacher training finished - beginning unlabelled prediction', '\n'*3)

    ### Predict lables for the unseen samples
    # Load unlabelled data
    # AutoQADataset
    args_new = args
    args_new.do_eval = False
    args_new.do_train = False
    args_new.do_predict = True
    args_new.task = 'go/third_party'
    unlabelled = AutoQADataset(args_new, 'ads', tokenizer)

    # Predict unlabelled data
    callbacks = [CustomFlowCallback]
    predictor = CustomTrainer(
        teacher,
        args=args_new,
        # eval_dataset=unlabelled,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        callbacks=callbacks,
        compute_metrics=compute_metrics,
    )        
    outputs = predictor.predict(test_dataset=unlabelled)
    preds = np.argmax(outputs.predictions, axis=1)
    with open(args.run_name + '/teacher_preds.txt', 'w') as f:
        for p in preds:
            f.write(str(p)+'\n')
    
    ### Train with combined dataset
    # Update unlablled data with predicted labels
    for sample, pred in zip(unlabelled.data, preds):
        setattr(sample, 'labels', pred)

    # Combine training datasets
    datasets['train'].data += unlabelled.data
    shuffle(datasets['train'].data)

    print('\n'*3, 'Prediction phase finished - beginning training on both', '\n'*3)
   
    # Train on both
    args.do_train = True   # TODO
    args.do_eval = True   # TODO
    args.do_predict = True   # TODO
    student, tokenizer, _ = classification_loop(args, datasets=datasets)


if __name__ == '__main__':
    main()