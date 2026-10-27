auto_scale_lr = None
crop_size = (
    512,
    512,
)
custom_hooks = [
    dict(type='ProgressHook'),
]
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'common',
    ])
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = 'data/ade/ADEChallengeData2016'
dataset_type = 'ADE20KDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        max_keep_ckpts=1,
        save_best='mIoU',
        save_last=True,
        save_optimizer=True,
        type='CheckpointHook'),
    logger=dict(interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True)
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='ReLU'),
        align_corners=False,
        channels=64,
        deploy=False,
        in_channels=3,
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_blocks_per_stage=[
            4,
            4,
            [
                5,
                4,
            ],
            [
                5,
                4,
            ],
            [
                2,
                2,
            ],
        ],
        ppm_channels=128,
        type='GCNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        act_cfg=dict(inplace=True, type='ReLU'),
        align_corners=False,
        channels=128,
        dropout_ratio=0.0,
        in_channels=256,
        loss_decode=[
            dict(loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
            dict(loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        ],
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=2,
        type='GCNetHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
num_classes = 2
optim_wrapper = dict(
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.001, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(custom_keys=dict(norm=dict(decay_mult=0.0))),
    type='AmpOptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(begin=0, by_epoch=True, end=2, start_factor=0.01, type='LinearLR'),
    dict(
        T_max=118,
        begin=2,
        by_epoch=True,
        end=120,
        eta_min=1e-06,
        type='CosineAnnealingLR'),
]
randomness = dict(diff_rank_seed=False, seed=42)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(img_path='images/test', seg_map_path='masks/test'),
        data_root='/root/autodl-tmp/work/cea_public_20260915/data/RiceSEG',
        img_suffix='.jpg',
        metainfo=dict(classes=(
            'background',
            'panicle',
        )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='.png',
        type='BaseSegDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2048,
        512,
    ), type='Resize'),
    dict(reduce_zero_label=True, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_epochs=120, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(img_path='images/train', seg_map_path='masks/train'),
        data_root='/root/autodl-tmp/work/cea_public_20260915/data/RiceSEG',
        img_suffix='.jpg',
        metainfo=dict(classes=(
            'background',
            'panicle',
        )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(direction='vertical', prob=0.3, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='.png',
        type='BaseSegDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=True, type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            2048,
            512,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        512,
        512,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(img_path='images/val', seg_map_path='masks/val'),
        data_root='/root/autodl-tmp/work/cea_public_20260915/data/RiceSEG',
        img_suffix='.jpg',
        metainfo=dict(classes=(
            'background',
            'panicle',
        )),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='.png',
        type='BaseSegDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/root/autodl-tmp/work/segmentation_comparisons_v2_20261011/runs/RiceSEG_GCNet_M_seed42'
