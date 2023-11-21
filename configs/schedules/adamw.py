# optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=1e-4))

# learning policy
warmup_epochs = 2  # about 10000 iterations for ImageNet-1k
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=warmup_epochs,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-5,
        by_epoch=True,
        begin=warmup_epochs)
]

# train, val, test setting
train_cfg = dict(type='EpochBasedTrainLoop', by_epoch=True, max_epochs=300, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(enable=False,base_batch_size=32)
