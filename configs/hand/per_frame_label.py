DURATION = 8
NUM_CLASSES = 3
ONLY_LAST = False


# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type="HandCNNLSTM",
        num_frames=DURATION,
    ),
    # neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LSTMHead',
        num_classes=NUM_CLASSES,
        only_last=ONLY_LAST,
        loss=dict(type='FocalLoss', loss_weight=1.0),
    ))

# dataset settings
dataset_type = 'HandSlideDataset'
train_pipeline = [
    dict(type='LandmarkToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=['landmark', 'label', 'src_depth_paths'])
]
test_pipeline = [
    dict(type='LandmarkToTensor', keys=['img']),
    dict(type='Collect', keys=['img'], meta_keys=['landmark', 'label', 'src_depth_paths'])
]


def gen_all_slide(root_dir, test_mode):
    import glob
    import os
    infer_result_dirs = glob.glob(os.path.join(root_dir, "**", "merge_result"), recursive=True)
    src_dirs = sorted([os.path.dirname(d) for d in infer_result_dirs])
    # fixme 写死防止空数据集
    src_dirs = list(filter(lambda p: "/up" in p or "/down" in p, src_dirs))
    return [gen_sub_data(d, test_mode) for d in src_dirs]


def gen_sub_data(src_dir, test_mode):
    return \
    dict(
    type=dataset_type, 
    data_prefix=src_dir,
    pipeline=train_pipeline if not test_mode else test_pipeline,
    duration=DURATION,
    num_keypoint=16,
    test_mode=test_mode,
    gt_per_frame=not ONLY_LAST) 


data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    train_dataloader=dict(shuffle=True),
    train=gen_all_slide("/data/dataset/hand/backup/slide/", False),
    val=gen_all_slide("/data/dataset/hand/backup/slide/", True),
    test=gen_all_slide("/data/dataset/hand/backup/slide/", True),
        )
# optimizer
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='AdamW', lr=1e-3)


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

evaluation = dict(
    interval=1, metric=['precision', 'recall', 'f1_score'],
    metric_options=dict(
        average_mode="none",
        thrs=(0., )
    ),
    rule="custom",
    save_best='auto')  # save the checkpoint with highest accuracy


dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

work_dir = "./work_dirs/hand/per_frame_label"