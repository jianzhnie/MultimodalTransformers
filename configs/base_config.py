# optimizer
optim_wrapper = dict(optimizer=dict(type="AdamW", lr=0.001, weight_decay=1e-4))

# learning policy
warmup_epochs = 2  # warmup_for first 2 epoch
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type="LinearLR",
        start_factor=1e-3,
        by_epoch=True,
        end=warmup_epochs,
        # update by iter
        convert_to_iter_based=True,
    ),
    # main learning rate scheduler
    dict(type="CosineAnnealingLR", eta_min=1e-5, by_epoch=True, begin=warmup_epochs),
]

# train, val, test setting
train_cfg = dict(
    type="EpochBasedTrainLoop", by_epoch=True, max_epochs=300, val_interval=1
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(enable=False, base_batch_size=32)

# defaults to use registries in mmpretrain
default_scope = "lmmtrain"

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type="IterTimerHook"),
    # print log every 100 iterations.
    logger=dict(type="LoggerHook", interval=100),
    # enable the parameter scheduler.
    param_scheduler=dict(type="ParamSchedulerHook"),
    # save checkpoint per epoch.
    checkpoint=dict(type="CheckpointHook", interval=1),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type="DistSamplerSeedHook"),
    # validation results visualization, set True to enable it.
    visualization=dict(type="VisualizationHook", enable=False),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend="nccl"),
)

# set visualizer
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="UniversalVisualizer", vis_backends=vis_backends, name="visualizer"
)
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
# set log level
log_level = "INFO"
# load from which checkpoint
load_from = None
# whether to resume training from the loaded checkpoint
resume = False
# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)
