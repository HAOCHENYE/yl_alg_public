# fp16 = dict(loss_scale=512.)
dataset_type = 'SegDatasets'
data_root = '/usr/videodate/dataset/human_segmentation/'
base_lr = 0.04
warmup_iters = 400

trainer = dict(type="HumanSegTrain")
tester = dict(type="HumanSegTest")

model = dict(
    type='ZxsSeg',
    video=False,
    backbone=dict(
        type='GeneralSandNet',
        in_channels=3,
        stem_channels=16,
        stage_channels=[32, 32, 48, 64],
        block_per_stage=(1, 3, 6, 3),
        expansion=[1, 3, 4, 4],
        num_out=4),
    neck=dict(type='ZxsNeck'),
    seg_head=dict(
        type='ZxsHead',
        dynamic_ssim=True,
        ))

train_pipline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadMaskAnnotations'),
            dict(type='RandomFlip',
                 flip_ratio=0.5,
                 flip_keys=['img', 'gt_mask']
                 ),
            dict(type='RandomCrop',
                 crop_size=(0.6, 0.6),
                 crop_type='relative_range',
                 crop_keys=['img', 'gt_mask']),
            dict(type='RandomAffine',
                 border_val=[0, 0, 0],
                 scaling_ratio_range=[0.9, 1.1],
                 affine_pipelines=[dict(
                     type='MaskAffine',
                 )]),
            dict(type='TrackMaskAug',
                 video=False,
                 prior_aug_pipelines=[
                     dict(type='RandomAffine',
                          scaling_ratio_range=[0.95, 1.05],
                          max_translate_ratio=0.02,
                          max_rotate_degree=5,
                          border_val=[0, 0, 0],
                          affine_pipelines=[
                              dict(type='MaskAffine'),
                              dict(type='ThinPlateSpline')])]),
            dict(type="ResizeImage",
                 pad_val=0,
                 resize_keys=['img', 'gt_mask', 'ori_img']
                 ),
            dict(type='ColorJitter'),
            dict(type='NoiseImage'),
            dict(type='RandomRadiusBlur'),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[128, 128, 128],
                to_rgb=True),
            dict(type='Formatting',
                 collect_key=["gt_mask", "img", 'ori_img']),
        ]

val_pipline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadMaskAnnotations'),
            dict(type="ResizeImage",
                 img_scale=(352, 352),
                 pad_val=0,
                 resize_keys=['img', 'gt_mask']
                 ),
            dict(
                type='Normalize',
                mean=(127.5, 127.5, 127.5),
                std=(128, 128, 128),
                to_rgb=True),
            dict(type='Formatting', collect_key=["img", "gt_mask"]),
        ]

test_pipelines = [
            dict(type='LoadImageFromFile'),
            dict(type="ResizeImage",
                 img_scale=(384, 384),
                 pad_val=0,
                 resize_keys=['img']
                 ),
            dict(
                type='Normalize',
                mean=(127.5, 127.5, 127.5),
                std=(128, 128, 128),
                to_rgb=True),
            dict(type='Formatting', collect_key=["img"]),
        ]


data = dict(
    train=dict(
        dataset=dict(type=dataset_type,
                     ann_file=data_root + "train_seg.json",
                     img_prefix=data_root,
                     classes=['person'],
                     pipeline=train_pipline,
                     filter_empty_gt=True),
        batch_sampler_cfg=dict(type="YoloBatchSampler", drop_last=False, input_dimension=(320, 320)),
        batch_size=16,
        num_workers=8),

    val=dict(
        dataset=dict(type=dataset_type,
                     ann_file=data_root + "val_seg.json",
                     img_prefix=data_root,
                     classes=['person'],
                     pipeline=val_pipline,
                     filter_empty_gt=True),
        pipeline=val_pipline,
        batch_size=8,
        num_workers=8),
    test=dict(
        dataset=dict(type=dataset_type,
                     ann_file=data_root + "val_seg.json",
                     img_prefix=data_root,
                     classes=['person'],
                     pipeline=test_pipelines,
                     filter_empty_gt=True),
        pipeline=test_pipelines,
        filter_empty_gt=True,
        batch_size=8,
        num_workers=8),
        )

optimizer = dict(type='AdaBelief')
# optimizer = dict(type='SGD', lr=base_lr)
# optimizer = dict(type='ACProp', lr=base_lr)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-7,
    warmup='linear',
    warmup_iters=warmup_iters,
    warmup_ratio=0.001)


total_epochs = 120
# learning policy

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

custom_hooks = [dict(type='SyncRandomSizeHook', ratio_range=(7, 11))]
evaluation = dict(interval=5)
evaluator = dict(type='SegEvaluator', data_type='tensor')

device_ids = range(0, 2)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './tasks/semantic_segmentation/work_dirs/zxs_dynamic_head'
load_from = None
# resume_from = '../tasks/semantic_segmentation/work_dirs/zxs_seg_a1/latest.pth'
resume_from = './tasks/semantic_segmentation/work_dirs/zxs_seg_3c/latest.pth'
workflow = [('train', 1)]
gpu_ids = range(0, 2)
