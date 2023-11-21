import sys

import torch
from torch.utils.data import DataLoader, random_split

from mmengine.model import BaseModel
from mmengine.optim import AmpOptimWrapper
from mmengine.runner import Runner
from transformers import BertTokenizer

sys.path.append("../")
from lmms.datasets.clip_dataset import Flikr8kDataset
from lmms.models.clip.config import ClipConfig as cfg
from lmms.models.clip.modeling_clip import CLIPModel


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
        outputs = self.model(
            input_ids,
            attention_mask,
            images,
            caption,
        )
        if mode == "loss":
            return {"loss": outputs[0]}


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
    optim_wrapper = dict(
        type="OptimWrapper",
        # 如果你想要使用 BF16，请取消下面一行的代码注释
        # dtype='bfloat16',  # 可用值： ('float16', 'bfloat16', None)
        optimizer=dict(
            type="AdamW",
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        ),
        # 累加 4 次参数更新一次
        accumulative_counts=4
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    optim_wrapper = AmpOptimWrapper(optimizer=optimizer, dtype="float16")

    # learning policy
    warmup_epochs = 1  # about 10000 iterations for ImageNet-1k
    param_scheduler = [
        # warm up learning rate scheduler
        dict(
            type="LinearLR",
            start_factor=1e-3,
            by_epoch=True,
            end=warmup_epochs,
            convert_to_iter_based=True,
        ),
        # main learning rate scheduler
        dict(
            type="CosineAnnealingLR", eta_min=1e-5, by_epoch=True, begin=warmup_epochs
        ),
    ]
    # configure default hooks
    default_hooks = dict(
        logger=dict(type="LoggerHook", interval=10),
    )
    visualizer = dict(
        type="Visualizer",
        vis_backends=[
            dict(type="WandbVisBackend", init_kwargs=dict(project="train_clip"))
        ],
    )
    env_cfg = dict(
        # whether to enable cudnn benchmark
        cudnn_benchmark=False,
        # set multi process parameters
        mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
        # set distributed parameters
        dist_cfg=dict(backend="nccl"),
    )
    runner = Runner(
        model=clip_model,
        train_dataloader=train_loader,
        env_cfg=env_cfg,
        optim_wrapper=optim_wrapper,
        param_scheduler=param_scheduler,
        default_hooks=default_hooks,
        train_cfg=dict(by_epoch=True, max_epochs=10, val_interval=1),
        default_scope="lmmtrain",
        visualizer=visualizer,
        work_dir="/home/robin/work_dir/llm/MultimodalTransformers/work_dir/clip_model",
    )
    runner.train()


if __name__ == "__main__":
    main()
