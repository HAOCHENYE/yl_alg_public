# fp16 = dict(loss_scale=512.)
dataset_type = 'WiderCocoDataset'
data_root = '/usr/videodate/yehc/ImageDataSets/WIDERFACE/'
base_lr = 0.02
warmup_iters = 500

trainer = dict(type="FaceDetectTrain")
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='VGGNet',
        stem_channels=32,
        stage_channels=(32, 64, 96, 128),
        block_per_stage=(1, 3, 8, 6),
        kernel_size=[3, 3, 3, 3],
        conv_cfg=dict(type="RepVGGConv"),
        num_out=3,
    ),
    neck=dict(
        type='RetinaFaceNeck',
        in_channels=[64, 96, 128],
        out_channels=64,
        ),
    bbox_head=dict(
        type='RetinaFace',
        in_channels=64,
        ))

train_pipline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            # dict(type='RandomFlip',
            #      flip_ratio=0.5,
            #      flip_bboxes=True,
            #      flip_landmarks=True),
            # dict(type='RandomCrop',
            #      crop_size=(0.3, 0.3),
            #      crop_type='relative_range',
            #      crop_landmarks=True,
            #      keep_bboxes_center_in=True),
            dict(type='ColorJitter'),
            dict(type="ResizeImage", img_scale=(640, 640), pad_val=127.5),
            dict(type="ResizeBboxes", bbox_clip_border=False),
            dict(type="ResizeLandMarks", bbox_clip_border=False),
            dict(
                type='Normalize',
                mean=127.5,
                std=128,
                to_rgb=True),
            dict(type='Formatting',
                 pad_cfg=dict(key=["gt_bboxes", "gt_labels", "gt_landmarks"], pad_num=2000),
                 collect_key=["gt_bboxes", "gt_labels", "gt_landmarks", "img"]),
        ]

val_pipline = [
            dict(type='LoadImageFromFile'),
            # dict(type="ResizeImage", img_scale=(1200, 1200), pad_val=127.5, keep_ratio=False),
            # dict(type='Pad', pad_to_square=True, pad_val=127.5),
            dict(
                type='Normalize',
                mean=127.5,
                std=128,
                to_rgb=True),
            dict(type='Formatting',
                 collect_key=["img"]),
        ]
data = dict(
    train=dict(
        dataset=dict(type=dataset_type,
                     ann_file=data_root + "wider_face_train_annot_coco_style_landmark.json",
                     img_prefix=data_root + 'WIDER_train/images',
                     classes=['face'],
                     pipeline=train_pipline,
                     filter_empty_gt=True),
        batch_size=32,
        num_workers=4),

    val=dict(
        dataset=dict(type=dataset_type,
                     ann_file=data_root + "wider_face_train_annot_coco_style_landmark.json",
                     img_prefix=data_root + 'WIDER_val/images',
                     classes=['face'],
                     pipeline=val_pipline,
                     filter_empty_gt=True),
        pipeline=val_pipline,
        filter_empty_gt=True),
    test=dict(
        dataset=dict(type=dataset_type,
                     ann_file=data_root + "wider_face_train_annot_coco_style_landmark.json",
                     img_prefix=data_root + 'WIDER_val/images',
                     classes=['face'],
                     pipeline=val_pipline,
                     filter_empty_gt=True),
        pipeline=val_pipline,
        filter_empty_gt=True),
        )


evaluation = dict(interval=100, metric='bbox', classwise=True)
optimizer = dict(type='SGD', lr=base_lr, momentum=0.937, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    min_lr=base_lr/1000,
    warmup='linear',
    warmup_iters=warmup_iters,
    warmup_ratio=0.001)


total_epochs = 24
# learning policy

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

# custom_hooks = [dict(type='SyncRandomSizeHook', ratio_range=(10, 24))]

device_ids = range(0, 2)
dist_params = dict(backend='nccl')
log_level = 'INFO'

work_dir = './face_detection/work_dirs/retinanet_try_singlescale_nocrop_noflip'
load_from = './face_detection/work_dirs/retina_face_10_08/latest.pth'
resume_from = None
# resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 2)

