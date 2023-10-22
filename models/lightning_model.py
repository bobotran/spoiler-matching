from transformers import AutoModelForSequenceClassification
import torchmetrics
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

class RelevanceRanker(pl.LightningModule):
    def __init__(self, model_name, lr, lr_interval, lr_step_type, weight_decay=1e-2, aggregator='max'):
        super(RelevanceRanker, self).__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        if aggregator == 'max':
            self.aggregator = torch.max
        elif aggregator == 'mean':
            self.aggregator = torch.mean
        else:
            raise ValueError()

        self.criterion = torch.nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 0.25, patience=4)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_mrr",
                "interval": self.hparams.lr_step_type,
                "frequency": self.hparams.lr_interval,
                "strict": False,
            },
        }

    def forward(self, **inputs):
        return self.model(**inputs).logits

    def training_step(self, train_batch, batch_idx):
        train_input, train_label = train_batch

        logits = self.forward(**train_input)

        loss = self.criterion(logits, train_label)

        self.log('ce_loss', loss, on_step=True, on_epoch=False, 
            prog_bar=True, logger=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        val_input, target, indexes, summary_indexes = val_batch

        logits = self.forward(**val_input)
        probs = F.softmax(logits, dim=1)

        return probs[:,1], target, indexes, summary_indexes

    def validation_epoch_end(self, validation_step_outputs):
        self.log('val_mrr', self.compute_mrr(validation_step_outputs), 
                prog_bar=True, logger=True)

    def test_step(self, test_batch, batch_idx):
        test_input, target, indexes, summary_indexes = test_batch

        logits = self.forward(**test_input)
        probs = F.softmax(logits, dim=1)

        return probs[:,1], target, indexes, summary_indexes

    def test_epoch_end(self, test_step_outputs):
        self.log('test_mrr', self.compute_mrr(test_step_outputs), 
                prog_bar=True, logger=True)

    def compute_mrr(self, outputs):
        '''Args:
            outputs: Input to validation_epoch_end or test_epoch_end
        '''
        # Sort results by comment and summary
        results = {}
        truth = {}
        for probs, targets, comment_indexes, summary_indexes in outputs:
            for i in range(len(probs)):
                prob, target, comment_idx, summary_idx = probs[i], targets[i], comment_indexes[i].item(), summary_indexes[i].item()
                if comment_idx not in results:
                    results[comment_idx] = {}
                if summary_idx not in results[comment_idx]:
                    results[comment_idx][summary_idx] = []
                results[comment_idx][summary_idx].append(prob)
                truth[(comment_idx, summary_idx)] = target
        # Aggregate results over summary and compute rr
        rrs = []
        for comment_idx in results:
            preds, targets = [], []
            for summary_idx in results[comment_idx]:
                passage_probs = torch.stack(results[comment_idx][summary_idx])
                preds.append(self.aggregator(passage_probs))
                targets.append(truth[(comment_idx, summary_idx)])
            preds, targets = torch.stack(preds), torch.stack(targets)
            rrs.append(torchmetrics.functional.retrieval_reciprocal_rank(preds, targets))
        
        if len(rrs) == 0: # Resuming from a checkpoint
            return -1.0
        return torch.mean(torch.stack(rrs))