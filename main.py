import json
import os

import fire
import torch
import yaml

from model import TFModel
from train import train_model
from utils import create_folder, fix_random_seed

print(torch.__version__)


class Config:
    """
    This is the configuration class to store the configuration of a TFModel. It is used to
    instantiate a model according to the specified arguments, defining the model architecture.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def make_scheduler(optimizer, config):
    if config.schedule == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    elif config.schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config.num_epoch
        )
    elif config.schedule == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.lr,  # Peak learning rate
            total_steps=config.num_epoch,
            pct_start=0.1,  # Percentage of training to increase LR
            anneal_strategy="cos",
            div_factor=25.0,  # Initial LR = max_lr/div_factor
            final_div_factor=10000.0,  # Final LR = max_lr/(div_factor*final_div_factor)
        )
    return scheduler


def main(**kwargs):
    with open("./config.yaml", "r") as file:
        config_args = yaml.safe_load(file)
    for k, v in kwargs.items():
        if k not in config_args:
            print(f"Warning: {k} is not supported!")
        if v != config_args[k]:
            print(f"{k} is overloaded from {config_args[k]} to {v}")
            config_args[k] = v
    config = Config(**config_args)

    fix_random_seed(config.seed, reproduce=True)
    create_folder(config.out_dir)
    print(config.__dict__)
    json.dump(config.__dict__, open(os.path.join(config.out_dir, "config.json"), "w"))

    model = TFModel(config).to(config.device)

    ## save init model
    out_path = os.path.join(config.out_dir, "ckpt_0.pt")
    torch.save(model.state_dict(), out_path)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config.wd,
    )

    scheduler = make_scheduler(optimizer, config)

    model, err_arr, err_arr_json = train_model(
        model=model,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    ## save model
    out_path = os.path.join(config.out_dir, "ckpt.pt")
    torch.save(model.state_dict(), out_path)

    json.dump(
        err_arr_json,
        open(os.path.join(config.out_dir, "err_arr.json"), "w"),
    )


if __name__ == "__main__":
    fire.Fire(main)
