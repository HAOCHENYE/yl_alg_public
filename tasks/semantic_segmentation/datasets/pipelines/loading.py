from general_datasets import PIPELINES
import mmcv
import os.path as osp


@PIPELINES.register_module()
class LoadMaskAnnotations:
    """Load multiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 color_type='grayscale',
                 file_client_args=dict(backend='disk')):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.color_type = color_type

    def _load_mask_file(self, results):
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['ann_info']['gt_maskfile'])
        else:
            filename = results['ann_info']['gt_maskfile']

        img_bytes = self.file_client.get(filename)
        gt_mask = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        results['gt_mask'] = gt_mask
        results['img_fields'].append('gt_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        results = self._load_mask_file(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_landmark}, '
        return repr_str


