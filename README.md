# BaseNLP

This repo contains skeleton code for NLP training loops. This repo This should help to re-use code between projects and streamline the development process. To initialise an NLP project, fork this repo and add functionality at the various specified locations.

This repo is only supposed to be a simplified version of the Huggingface Transformers repo. For any questions about functionality, usage etc. [consult the original repo](https://github.com/huggingface/transformers).

## Setup

I would recommend forking this repo to begin your new workflow.

- Code only tested for >= python3.6
- This should be run from a virtual environment e.g.:

```
virtualenv -p python3 <path to venv>
. <path to venv>/bin/activate
```

- Install requirements within the virtual environment:

```
pip install -r requirements.txt
```

# Classification

This section contains base code for training and/or evaluating classification models.

- Launch using `classification.py`
    - Training/fine-tuning: ```python classification.py train```
    - Evaluate on test set: ```python classification.py test```
- These commands load full arguments from .yaml config files within [config/classification](config/classification)
- These commands run using a demo dataset included within [data/snli_small](data/snli_small). 
    - This is a natural language inference NLP task, i.e. determine if sentence 2 is entailed, contradicted or neither by sentence 1. This is a 3-way classification problem
    - To add your own dataset, you will need to write new dataloaders in [scr/classification/data.py](scr/classification/data.py) and [scr/classification/utils.py](scr/classification/utils.py)
- To implement more customised behaviour, edit the files within [src/classification](src/classification). E.g., to implement custom training behaviour, edit [src/classification/custom_trainer.py](src/classification/custom_trainer.py)

# Sequence-to-sequence

This section contains base code for training and/or evaluating sequence-to-sequence models. These are generally more complex to create and run than the classification case above. This code is largely drawn from [the Huggingface seq2seq examples repo](https://github.com/huggingface/transformers/tree/master/examples/seq2seq) (Note: this repo is changed regularly so you may need to align versions to see source).

- Launch training/fine-tuning with ```python finetune_seq2seq.py```
- Evaluate on test set: ```python eval_seq2seq.py```
- These commands load full arguments from .yaml config files within [config/seq2seq](config/seq2seq)
- These commands run using a demo dataset included within [data/cnn_dm_small](data/cnn_dm_small)
    - This is a summarization task involving news articles from the DailyMail and CNN
    - To add your own dataset, you will need to write new dataloaders in [scr/seq2seq/utils.py](scr/seq2seq/utils.py)
- To implement more customised behaviour, edit the files within [src/seq2seq](src/seq2seq). E.g., to implement custom training behaviour, edit [src/seq2seq/custom_trainer.py](src/seq2seq/custom_trainer.py)

