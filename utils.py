import numpy as np
import json
import copy
import os
import random
import mmcv
from mmcv.runner.hooks import HOOKS, Hook, LoggerHook
from mmcv.runner.dist_utils import master_only
from mmcv import Config, ConfigDict
import numbers


def search_and_modify_cfg(cfg, key, value):
    if isinstance(cfg, list):
        for e in cfg:
            search_and_modify_cfg(e, key, value)
    elif isinstance(cfg, (Config, ConfigDict)):
        for k, v in cfg.items():
            if k == key:
                cfg[k] = value
            else:
                search_and_modify_cfg(v, key, value)


class UserStop(Exception):
    pass


def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert mmcv.is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals


def register_mmlab_modules():
    # Define custom hook to stop process when user uses stop button and to save last checkpoint
    @HOOKS.register_module(force=True)
    class CustomHook(Hook):
        # Check at each iter if the training must be stopped
        def __init__(self, stop, output_folder, emit_step_progress):
            self.stop = stop
            self.output_folder = output_folder
            self.emit_step_progress = emit_step_progress

        def after_epoch(self, runner):
            self.emit_step_progress()

        def after_train_iter(self, runner):
            # Check if training must be stopped and save last model
            if self.stop():
                runner.save_checkpoint(self.output_folder, "latest.pth", create_symlink=False)
                raise UserStop

    @HOOKS.register_module(force=True)
    class CustomMlflowLoggerHook(LoggerHook):
        """Class to log metrics and (optionally) a trained model to MLflow.
        It requires `MLflow`_ to be installed.
        Args:
            interval (int): Logging interval (every k iterations). Default: 10.
            ignore_last (bool): Ignore the log of last iterations in each epoch
                if less than `interval`. Default: True.
            reset_flag (bool): Whether to clear the output buffer after logging.
                Default: False.
            by_epoch (bool): Whether EpochBasedRunner is used. Default: True.
        .. _MLflow:
            https://www.mlflow.org/docs/latest/index.html
        """

        def __init__(self,
                     log_metrics,
                     interval=10,
                     ignore_last=True,
                     reset_flag=True,
                     by_epoch=True):
            super(CustomMlflowLoggerHook, self).__init__(interval, ignore_last,
                                                         reset_flag, by_epoch)
            self.log_metrics = log_metrics

        @master_only
        def log(self, runner):
            metrics = {}

            for var, val in runner.log_buffer.output.items():
                if var in ['time', 'data_time']:
                    continue
                tag = f'{var}/{runner.mode}'
                if isinstance(val, numbers.Number):
                    metrics[tag] = val
            metrics['learning_rate'] = runner.current_lr()[0]
            metrics['momentum'] = runner.current_momentum()[0]
            self.log_metrics(metrics, step=runner.iter)


def poly_area(pts):
    x = pts[::2]
    y = pts[1::2]
    return 0.5 * (np.dot(x, np.roll(y, 1)) - np.dot(np.roll(x, 1), y))


def polygone_to_bbox_xywh(pts):
    """
    :param pts: list of coordinates with xs,ys respectively even,odd indexes
    :return: array of the bounding box xywh
    """
    x = np.min(pts[0::2])
    y = np.min(pts[1::2])
    w = np.max(pts[0::2]) - x
    h = np.max(pts[1::2]) - y
    return [x, y, w, h]


def fill_dict(json_dict, sample, id, id_annot):
    json_dict['images'].append({'file_name': sample['filename'],
                                'height': sample['height'],
                                'width': sample['width'],
                                'id': id})
    for annot in sample['annotations']:
        if 'bbox' in annot:
            annot_to_write = {}
            annot_to_write['iscrowd'] = 0
            annot_to_write['category_id'] = annot["category_id"]
            bbox = annot['bbox']
            x, y, w, h = bbox
            annot_to_write['bbox'] = bbox
            if "segmentation_poly" in annot:
                if len(annot["segmentation_poly"]):
                    annot_to_write["segmentation"] = annot["segmentation_poly"]
                    annot_to_write["area"] = sum([poly_area(pts) for pts in annot["segmentation_poly"]])
                else:
                    annot_to_write['segmentation'] = [[x, y, x + w, y, x + w, y + h, x, y + h]]
                    annot_to_write['area'] = w * h
            else:
                annot_to_write['segmentation'] = [[x, y, x + w, y, x + w, y + h, x, y + h]]
                annot_to_write['area'] = w * h
            annot_to_write['image_id'] = id
            annot_to_write['id'] = id_annot
            id_annot += 1
            json_dict['annotations'].append(annot_to_write)

    return id_annot


def prepare_dataset(ikdata, save_dir, split_ratio):
    dataset_path = os.path.join(save_dir, 'dataset')
    train_file = os.path.join(dataset_path, 'instances_train.json')
    test_file = os.path.join(dataset_path, 'instances_test.json')

    os.makedirs(dataset_path, exist_ok=True)
    print("Preparing dataset...")

    images = ikdata['images']
    n = len(images)
    train_idx = random.sample(range(n), int(n * split_ratio))
    json_train = {'images': [],
                  'categories': [{"id": k, "name": v} for k, v in ikdata['metadata']['category_names'].items()],
                  'annotations': []}
    json_test = copy.deepcopy(json_train)
    id_annot = 0
    for id, sample in enumerate(images):
        if id in train_idx:
            id_annot = fill_dict(json_train, sample, id, id_annot)
        else:
            id_annot = fill_dict(json_test, sample, id, id_annot)

    with open(train_file, 'w') as f:
        json.dump(json_train, f)
    with open(test_file, 'w') as f:
        json.dump(json_test, f)

    print("Dataset prepared!")
