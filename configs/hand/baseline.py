# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

# dataset settings
dataset_type = 'HandSlideDataset'
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False)
train_pipeline = [
    # dict(type='RandomCrop', size=32, padding=4),
    # dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['embedding']),
    dict(type='ToTensor', keys=['target']),
    dict(type='Collect', keys=['embedding', 'target'], meta_keys=['landmark', 'label', 'src_depth_paths'])
]
test_pipeline = [
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['embedding']),
    dict(type='Collect', keys=['embedding', 'target'], meta_keys=['landmark', 'label', 'src_depth_paths'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type, 
        data_prefix="/data/dataset/hand/backup/slide/2022_11_28/guoqihang/down/",
        pipeline=train_pipeline,
        duration=8,
        num_keypoint=16,
        test_mode=False),
    val=dict(
        type=dataset_type, 
        data_prefix="/data/dataset/hand/backup/slide/2022_11_28/guoqihang/down/",
        pipeline=test_pipeline,
        duration=8,
        num_keypoint=16,
        test_mode=True),
    test=dict(
        type=dataset_type, 
        data_prefix="/data/dataset/hand/backup/slide/2022_11_28/guoqihang/down/",
        pipeline=test_pipeline,
        duration=8,
        num_keypoint=16,
        test_mode=True),
        )
# optimizer
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='AdamW', lr=1e-3)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[100, 150])
runner = dict(type='EpochBasedRunner', max_epochs=200)

# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
