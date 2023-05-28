import torch 
import torch.nn as nn
import segmentation_models_pytorch as smp
from data_load import load_dataset
from torch.optim import Adam, lr_scheduler
import lightning as L
import time

# My modules
from config import config, aux_params


'''def calculate_iou(pred_labels, true_labels):
    intersection = torch.logical_and(pred_labels, true_labels).sum()
    union = torch.logical_or(pred_labels, true_labels).sum()
    iou = intersection.item() / (union.item() + 1e-7)  # Add epsilon to avoid division by zero
    return iou


def convert_to_binary_mask(output):
    binary_mask = torch.where(output >= config["threshold"], torch.ones_like(output), torch.zeros_like(output))
    return binary_mask'''


'''class MBSegModel():
    def __init__(self):
        self.iou_history = {"train": [], "val": []}
        self.loss_history = {"train": [], "val": []}
        self.device = config["device"]
        self.loader = load_dataset()
        self.model = smp.Unet(model_params, aux_params=aux_params).to(self.device)
        self.criterion, self.optimizer, self.scheduler = self.define_optim(self.model)

    def fit():
        for i in range(config["num_epochs"]):


    def training_step(self):
        total_loss = 0
        total_iou = 0
        self.model.train()
        for i, (inputs, labels) in enumerate(self.loader['train']):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            self.optimizer.zero_grad()
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            preds = convert_to_binary_mask(outputs)
            iou = calculate_iou(preds, labels)
            total_loss += loss.item()
            total_iou += iou.item()
        self.scheduler.step()
        self.loss_history["train"].append(total_loss)
        self.iou_history["train"].append(total_iou)

    def validation_step(self):
        total_loss = 0
        total_iou = 0
        self.model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.loader['val']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                preds = convert_to_binary_mask(outputs)
                iou = calculate_iou(preds, labels)
                total_loss += loss.item()
                total_iou += iou.item()
        self.model.train()
        self.loss_history["val"].append(total_loss)
        self.iou_history["val"].append(total_iou)

    def save(self, path):
        torch.save(self.model, path)

    def save_onnx(self, path):
        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        torch.onnx.export(self.model, dummy_input, path)'''
 
class MBSegModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.outputs = None
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.model = smp.Unet(encoder_name=config["encoder_name"], 
                              encoder_weights=config["encoder_weights"],
                              in_channels=3,
                              classes=1)
        
    def forward(self, x):
        return self.model(x.float())

    def shared_step(self, batch, stage):
        inputs, labels = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        prob_mask = logits.sigmoid()
        pred_mask = (prob_mask > config["threshold"]).float()
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), labels.long(), mode="binary")
        self.outputs = {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        return {
            "loss": loss
        }

    def shared_epoch_end(self, stage):
        tp = self.outputs["tp"]
        fp = self.outputs["fp"]
        fn = self.outputs["fn"]
        tn = self.outputs["tn"]
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        '''metrics = {
            f"{stage}_loss": self.outputs["loss"],
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }  '''
        self.logger.experiment.add_scalar(f"{stage}_loss",
                                            self.outputs["loss"],
                                            self.current_epoch)
        self.logger.experiment.add_scalar(f"{stage}_per_image_iou",
                                            per_image_iou,
                                            self.current_epoch)
        self.logger.experiment.add_scalar(f"{stage}_dataset_iou",
                                            dataset_iou,
                                            self.current_epoch) 

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def on_train_epoch_end(self):
        return self.shared_epoch_end("train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def on_validation_epoch_end(self):
        return self.shared_epoch_end("valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def on_test_epoch_end(self):
        return self.shared_epoch_end("test")

    def configure_optimizers(self):
        optimizer = Adam(
                    self.parameters(), 
                    lr=config["learning_rate"], 
                    weight_decay=config["weight_decay"])
        exp_lr_scheduler = lr_scheduler.StepLR(
                    optimizer, 
                    step_size=config["scheduler_step"], 
                    gamma=config["scheduler_gamma"])
        scheduler_config = {
            "scheduler": exp_lr_scheduler,
            "interval": config["scheduler_interval"]
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config
        }


    









