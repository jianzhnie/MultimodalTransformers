import sys

import torch
from torch.utils.data import DataLoader, random_split

from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from mmengine.optim.scheduler import MultiStepLR
from mmengine.runner import Runner
from transformers import BertTokenizer

sys.path.append("../")
from lmms.datasets.clip_dataset import Flikr8kDataset
from lmms.models.clip.config import ClipConfig as cfg
from lmms.models.clip.model import CLIPModel


def split_data(dataset: Flikr8kDataset, val_split: float):
    train_length = int((1 - val_split) * len(dataset))
    val_length = len(dataset) - train_length
    train_dataset, val_dataset = random_split(
        dataset, lengths=[train_length, val_length]
    )
    return train_dataset, val_dataset


class MMClipModel(BaseModel):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids,
        attention_mask,
        images,
        caption,
        mode="loss",
    ):
        loss = self.model(
            input_ids,
            attention_mask,
            images,
            caption,
        )
        if mode == "loss":
            return {"loss": loss}


def main() -> None:
    tokenizer = BertTokenizer.from_pretrained(cfg.text_tokenizer_name)
    dataset = Flikr8kDataset(
        image_dir=cfg.image_path,
        tokenizer=tokenizer,
        target_size=cfg.target_size,
        max_length=cfg.temperature,
    )
    train_ds, val_ds = split_data(dataset, val_split=0.2)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    model = CLIPModel(
        image_encoder_alias=cfg.image_model_name,
        text_encoder_alias=cfg.text_model_name,
        image_encoder_pretrained=cfg.image_encoder_pretrained,
        text_encoder_pretrained=cfg.text_encoder_pretrained,
        image_encoder_trainable=cfg.image_encoder_trainable,
        text_encoder_trainable=cfg.text_encoder_trainable,
        image_embedding_dims=cfg.image_embedding_dims,
        text_embedding_dims=cfg.text_embedding_dims,
        projection_dims=cfg.projection_dims,
        dropout=cfg.dropout,
        temperature=cfg.temperature,
    ).to(cfg.device)

    clip_model = MMClipModel(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    optim_wrapper = OptimWrapper(optimizer)
    param_scheduler = MultiStepLR(optimizer, milestones=[5, 8], gamma=0.1)
    runner = Runner(
        model=clip_model,
        train_dataloader=train_loader,
        optim_wrapper=optim_wrapper,
        param_scheduler=param_scheduler,
        train_cfg=dict(by_epoch=True, max_epochs=10, val_interval=1),
        work_dir="/home/robin/work_dir/llm/MultimodalTransformers/work_dir/clip_model",
    )
    runner.train()


if __name__ == "__main__":
    main()
