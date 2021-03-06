# fp16 = dict(loss_scale=512.)
dataset_type = 'WiderCocoDataset'
data_root = '/usr/videodate/yehc/ImageDataSets/WIDERFACE/'
base_lr = 0.02
warmup_iters = 500

trainer = dict(type="FaceDetectTrain")
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='MobileNetV1',
        pretrained='mobilenet0.25_epoch_20.pth'
    ),
    neck=dict(
        type='FPN',
        in_channels_list=[64, 128, 256],
        out_channels=64,
        pretrained='mobilenet0.25_epoch_20.pth',
        ),
    bbox_head=dict(
        type='RetinaFace',
        in_channels=64,
        pretrained='mobilenet0.25_epoch_20.pth'
        ))

train_pipline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            # dict(type='RandomFlip', flip_ratio=0.5, flip_bboxes=True, flip_landmarks=True),
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
            # dict(type="ResizeImage", img_scale=(640, 640), pad_val=127.5, keep_ratio=False),
            # dict(type='Pad', size_divisor=32, pad_val=127.5),
            # dict(type='Pad', pad_to_square=True, pad_val=127.5),
            dict(
                type='Normalize',
                mean=127.5,
                std=128,
                to_rgb=False),
            dict(type='Formatting', collect_key=["img"]),
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

evaluation = dict(interval=1000, metric='bbox', classwise=True)
optimizer = dict(type='SGD', lr=base_lr, momentum=0.937, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    min_lr=base_lr/1000,
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

# custom_hooks = [dict(type='SyncRandomSizeHook', ratio_range=(16, 24))]

device_ids = range(0, 2)
dist_params = dict(backend='nccl')
log_level = 'INFO'

work_dir = './face_detection/work_dirs/retinaface_singlescale_crop_filp_sgd_noflip_nocrop_ori_anchor_gen'
load_from = None
resume_from = None
# resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 2)

