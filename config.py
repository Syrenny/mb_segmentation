import torch 


config = {
    "std": [0.229, 0.224, 0.225],
    "mean": [0.485, 0.456, 0.406],
    "num_epochs": 100,
    "learning_rate": 3e-4,
    "weight_decay": 8e-3,
    "scheduler_interval": "epoch",
    "scheduler_step": 1,
    "scheduler_gamma": 0.98,
    "threshold": 0.5, 
    "batch_size": 4,
    "encoder_name": "resnet50", 
    "encoder_weights": "imagenet",
    "data_dir": "massachusetts-buildings-dataset/tiff",
    "num_workers": 20
}

aux_params = dict(
    pooling='avg',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    activation='sigmoid',      # activation function, default is None
    classes=1                 # define number of output labels
)

