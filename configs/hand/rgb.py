DURATION = 20
NUM_CLASSES = 3
ONLY_LAST = True
NUM_KEYPOINTS = 21
SINGLE_FINGER = False

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type="HandCNNLSTM",
        num_frames=DURATION,
        feat_num=NUM_KEYPOINTS*2
    ),
    # neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LSTMHead',
        num_classes=NUM_CLASSES,
        only_last=ONLY_LAST,
        loss=dict(type='FocalLoss', loss_weight=1.0),
    ))

# dataset settings
dataset_type = 'HandSlideDatasetMaxTIOU'
train_pipeline = [
    # dict(type='LandmarkAddNoise', single_finger=SINGLE_FINGER),
    dict(type='LandmarkNormalize'),
    dict(type='LandmarkToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=['landmark', 'label', 'src_depth_paths'])
]
test_pipeline = [
    dict(type='LandmarkNormalize'),
    dict(type='LandmarkToTensor', keys=['img']),
    dict(type='Collect', keys=['img'], meta_keys=['landmark', 'label', 'src_depth_paths'])
]


def gen_all_slide(root_dir, test_mode):
    import glob
    import os
    infer_result_dirs = glob.glob(os.path.join(root_dir, "**", "up"), recursive=True) + glob.glob(os.path.join(root_dir, "**", "down"), recursive=True)
    src_dirs = sorted(set(os.path.dirname(d) for d in infer_result_dirs))
    print(test_mode, ':')
    [print(p) for p in src_dirs]
    print("-----------------------------------------")
    return [gen_sub_data(d, test_mode) for d in ["/data/dataset/hand/backup/slide_120fps/heshujie/", "/data/dataset/hand/backup/slide_120fps/zhangjuanjuan/",
                                                 "/data/dataset/hand/backup/slide_120fps/yufenni"]]
    # return [gen_sub_data(d, test_mode) for d in src_dirs]


def gen_sub_data(src_dir, test_mode):
    return \
    dict(
    type=dataset_type, 
    src_dir=src_dir,
    pipeline=train_pipeline if not test_mode else test_pipeline,
    duration=DURATION,
    num_keypoints=NUM_KEYPOINTS,
    single_finger=SINGLE_FINGER,
    test_mode=test_mode)


data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    train_dataloader=dict(shuffle=True),
    train=gen_all_slide("/data/dataset/hand/backup/slide_120fps/", False),
    val=gen_all_slide("/data/dataset/hand/backup/slide_120fps/", True),
    test=gen_all_slide("/data/dataset/hand/backup/slide_120fps/", True),
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
    rule="hand_slide",
    save_best='auto')  # save the checkpoint with highest accuracy


dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

work_dir = "./work_dirs/hand/p100"