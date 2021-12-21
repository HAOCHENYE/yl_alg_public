# fp16 = dict(loss_scale=512.)
dataset_type = 'SegDatasets'
data_root = '/usr/videodate/dataset/human_segmentation/'
base_lr = 0.04
warmup_iters = 400

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
        type='ZxsHead'
        ))

img_train_pipline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadMaskAnnotations'),
            dict(type='RandomFlip',
                 flip_ratio=0.5,
                 flip_keys=['img', 'gt_mask']
                 ),
            dict(type='RandomAffine',
                 border_val=[0, 0, 0],
                 scaling_ratio_range=[0.8, 1.2],
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
                           border_val=[0, 0, 0]),
                      dict(type='MotionBlur'),
                      dict(type='PersAffine'),
                      dict(type='ThinPlateSpline')]),
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

video_train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadMaskAnnotations'),
            dict(type='RandomFlip',
                 flip_ratio=0.5,
                 flip_keys=['img', 'gt_mask']
                 ),
            dict(type='RandomAffine',
                 border_val=[0, 0, 0],
                 scaling_ratio_range=[0.9, 1.1],
                 affine_pipelines=[dict(
                     type='MaskAffine',
                 )]),
            dict(type='TrackMaskAug',
                 prior_prob=0.5,
                 video=True,
                 prior_aug_pipelines=[dict(type='RandomAffine',
                                           scaling_ratio_range=[0.95, 1.05],
                                           max_translate_ratio=0.02,
                                           max_rotate_degree=5,
                                           border_val=[0, 0, 0]),
                                      dict(type='MotionBlur'),
                                      dict(type='PersAffine'),
                                      dict(type='ThinPlateSpline')]),
                 # prior_aug_pipelines=[dict(type='MotionBlur'),
                 #                      dict(type='WarpAffine'),
                 #                      dict(type='PersAffine'),
                 #                      dict(type='ThinPlateSpline')]),
            dict(type="ResizeImage",
                 pad_val=0,
                 resize_keys=['img', 'gt_mask', 'ori_img', 'prior_mask']
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
                 collect_key=["gt_mask", "img", 'ori_img', 'prior_mask']),
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


img_data = dict(
    train=dict(
        dataset=dict(type=dataset_type,
                     ann_file=data_root + "train_seg.json",
                     img_prefix=data_root,
                     classes=['person'],
                     pipeline=img_train_pipline,
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

video_data = dict(
    train=dict(
        dataset=dict(type=dataset_type,
                     ann_file=data_root + "train_seg.json",
                     img_prefix=data_root,
                     classes=['person'],
                     pipeline=video_train_pipeline,
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

img_optimizer = dict(type='AdaBelief')
img_optimizer_config = dict(grad_clip=None)
img_lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-7,
    warmup='linear',
    warmup_iters=warmup_iters,
    warmup_ratio=0.001)

img_total_epochs = 120

video_optimizer = dict(type='AdaBelief', lr=1e-3)
video_optimizer_config = dict(grad_clip=None)
video_lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-7,
    warmup='linear',
    warmup_iters=warmup_iters,
    warmup_ratio=0.001)
video_total_epochs = 60
# learning policy
video_cfg = dict(enabled=True,
                 data=video_data,
                 optimizer=video_optimizer,
                 total_epochs=video_total_epochs,
                 optimizer_config=video_optimizer_config,
                 lr_config=video_lr_config,
                 resume_from=None,
                 load_from=None)

img_cfg = dict(enabled=True,
               data=img_data,
               optimizer=img_optimizer,
               total_epochs=img_total_epochs,
               optimizer_config=img_optimizer_config,
               lr_config=img_lr_config,
               resume_from='./tasks/semantic_segmentation/work_dirs/zxs_seg_new_trainer/latest.pth',
               load_from=None)

trainer = dict(type="HumanVideoSegTrain",
               img_cfg=img_cfg,
               video_cfg=video_cfg
               )

tester = dict(type="HumanVideoSegTest",
              img_cfg=img_cfg,
              video_cfg=video_cfg,
              )
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

custom_hooks = [dict(type='SyncRandomSizeHook', ratio_range=(7, 11), iter_interval=1)]
evaluation = dict(interval=5)
evaluator = dict(type='SegEvaluator', data_type='tensor')

device_ids = range(0, 2)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './tasks/semantic_segmentation/work_dirs/zxs_seg_new_trainer'
load_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 2)
find_unused_parameters = True
