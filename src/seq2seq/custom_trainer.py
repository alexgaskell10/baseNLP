from transformers import Seq2SeqTrainer

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs):
        ''' Overwrites parent class for custom behaviour during training
        '''
        return super().training_step(model, inputs)

    def compute_loss(self, model, inputs, *args, **kwargs):
        ''' How the loss is computed by Trainer.
            Overwrites parent class for custom behaviour during training
        '''
        return super().compute_loss(model, inputs)

    def _compute_loss(self, model, inputs, *args, **kwargs):
        ''' Overwrites parent class for custom behaviour during training
        '''
        return super()._compute_loss(model, inputs)

    def prediction_step(self, model, inputs, *args, **kwargs):
        ''' Overwrites parent class for custom behaviour during prediction
        '''
        return super().prediction_step(model, inputs, *args, **kwargs)

    def log(self, *args):
        ''' Overwrites parent class for custom behaviour during training
        '''
        super().log(*args)
