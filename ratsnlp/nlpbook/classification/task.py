from transformers import PreTrainedModel
from transformers.optimization import AdamW
from ratsnlp.nlpbook.metrics import accuracy
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts
from ratsnlp.nlpbook.classification.arguments import ClassificationTrainArguments


class ClassificationTask(LightningModule):

    def __init__(self,
                 model: PreTrainedModel,
                 args: ClassificationTrainArguments,
    ):
        super().__init__()
        self.model = model
        self.args = args

    def configure_optimizers(self):
        if self.args.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.args.learning_rate)
        else:
            raise NotImplementedError('Only AdamW is Supported!')
        if self.args.lr_scheduler == 'cos':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        elif self.args.lr_scheduler == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def training_step(self, inputs, batch_idx):
        # outputs: SequenceClassifierOutput
        outputs = self.model(**inputs)
        preds = outputs.logits.argmax(dim=-1)
        labels = inputs["labels"]
        acc = accuracy(preds, labels)
        self.log("loss", outputs.loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("acc", acc, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return outputs.loss

    def validation_step(self, inputs, batch_idx):
        # outputs: SequenceClassifierOutput
        outputs = self.model(**inputs)
        preds = outputs.logits.argmax(dim=-1)
        labels = inputs["labels"]
        acc = accuracy(preds, labels)
        self.log("val_loss", outputs.loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return outputs.loss
