# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
import mmcv
import torch
import numpy as np
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor


def init_segmentor(config, checkpoint=None, device='cuda:0'):
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    # import pdb; pdb.set_trace()
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
            img = mmcv.imread(results['img'])
        elif isinstance(results['img'], np.ndarray):
            results['filename'] = None
            results['ori_filename'] = None
            img = results['img']
        else:
            raise TypeError(f"Unsupported image type: {type(results['img'])}")

        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_segmentor(model, img, twohands_list=None, cb_list=None):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        (list[Tensor]): The segmentation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    # normalize inputs to a list of images
    if isinstance(img, np.ndarray) and img.ndim == 4:
        imgs = [img[i] for i in range(img.shape[0])]
    elif isinstance(img, (list, tuple)):
        imgs = list(img)
    else:
        imgs = [img]

    # prepare data items
    data_list = [test_pipeline(dict(img=_img)) for _img in imgs]
    data = collate(data_list, samples_per_gpu=len(data_list))

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # inject optional meta keys
    if 'additional_channel' in cfg.keys():
        for b in range(len(data['img_metas'][0])):
            data['img_metas'][0][b]['additional_channel'] = cfg['additional_channel']
    if 'twohands_dir' in cfg.keys():
        for b in range(len(data['img_metas'][0])):
            data['img_metas'][0][b]['twohands_dir'] = cfg['twohands_dir']
    if 'cb_dir' in cfg.keys():
        for b in range(len(data['img_metas'][0])):
            data['img_metas'][0][b]['cb_dir'] = cfg['cb_dir']

    # attach in-memory auxiliary masks if provided
    if twohands_list is not None:
        assert len(twohands_list) == len(data['img_metas'][0]), 'twohands_list length must match batch size'
        for b in range(len(data['img_metas'][0])):
            data['img_metas'][0][b]['twohands_array'] = twohands_list[b]
    if cb_list is not None:
        assert len(cb_list) == len(data['img_metas'][0]), 'cb_list length must match batch size'
        for b in range(len(data['img_metas'][0])):
            data['img_metas'][0][b]['cb_array'] = cb_list[b]

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    return result


def show_result_pyplot(model,
                       img,
                       result,
                       palette=None,
                       fig_size=(15, 10),
                       opacity=0.5,
                       title='',
                       block=True):
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of segmentation
            map. If None is given, random palette will be generated.
            Default: None
        fig_size (tuple): Figure size of the pyplot figure.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        title (str): The title of pyplot figure.
            Default is ''.
        block (bool): Whether to block the pyplot figure.
            Default is True.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(
        img, result, palette=palette, show=False, opacity=opacity)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.title(title)
    plt.tight_layout()
    plt.show(block=block)
