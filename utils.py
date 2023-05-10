import numpy as np
import json
import copy
import os
import random
from mmengine.config import Config, ConfigDict

from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.hooks import LoggerHook
from mmengine.dist import master_only
from typing import Optional, Sequence, Union

DATA_BATCH = Optional[Union[dict, tuple, list]]

class UserStop(Exception):
    pass

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

        def _after_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[Union[Sequence, dict]] = None,
                        mode: str = 'train') -> None:
            # Check if training must be stopped and save last model
            if self.stop():
                runner.save_checkpoint(self.output_folder, "latest.pth")
                raise UserStop

    @HOOKS.register_module(force=True)
    class CustomLoggerHook(LoggerHook):
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
                     interval=10):
            super(CustomLoggerHook, self).__init__(interval=interval, log_metric_by_epoch=True)
            self.log_metrics = log_metrics

        def after_val_epoch(self,
                            runner,
                            metrics=None) -> None:
            """All subclasses should override this method, if they need any
            operations after each validation epoch.

            Args:
                runner (Runner): The runner of the validation process.
                metrics (Dict[str, float], optional): Evaluation results of all
                    metrics on validation dataset. The keys are the names of the
                    metrics, and the values are corresponding results.
            """
            tag, log_str = runner.log_processor.get_log_after_epoch(
                runner, len(runner.val_dataloader), 'val')
            runner.logger.info(log_str)
            if self.log_metric_by_epoch:
                # when `log_metric_by_epoch` is set to True, it's expected
                # that validation metric can be logged by epoch rather than
                # by iter. At the same time, scalars related to time should
                # still be logged by iter to avoid messy visualized result.
                # see details in PR #278.
                metric_tags = {k: v for k, v in tag.items() if 'time' not in k}
                runner.visualizer.add_scalars(
                    metric_tags, step=runner.epoch, file_path=self.json_log_path)
                self.log_metrics(tag, step=runner.epoch)
            else:
                runner.visualizer.add_scalars(
                    tag, step=runner.iter, file_path=self.json_log_path)
                self.log_metrics(tag, step=runner.iter + 1)

        def after_train_iter(self,
                             runner,
                             batch_idx: int,
                             data_batch=None,
                             outputs=None):
            """Record logs after training iteration.

            Args:
                runner (Runner): The runner of the training process.
                batch_idx (int): The index of the current batch in the train loop.
                data_batch (dict tuple or list, optional): Data from dataloader.
                outputs (dict, optional): Outputs from model.
            """
            # Print experiment name every n iterations.
            if self.every_n_train_iters(
                    runner, self.interval_exp_name) or (self.end_of_epoch(
                runner.train_dataloader, batch_idx)):
                exp_info = f'Exp name: {runner.experiment_name}'
                runner.logger.info(exp_info)
            if self.every_n_inner_iters(batch_idx, self.interval):
                tag, log_str = runner.log_processor.get_log_after_iter(
                    runner, batch_idx, 'train')
            elif (self.end_of_epoch(runner.train_dataloader, batch_idx)
                  and not self.ignore_last):
                # `runner.max_iters` may not be divisible by `self.interval`. if
                # `self.ignore_last==True`, the log of remaining iterations will
                # be recorded (Epoch [4][1000/1007], the logs of 998-1007
                # iterations will be recorded).
                tag, log_str = runner.log_processor.get_log_after_iter(
                    runner, batch_idx, 'train')
            else:
                return
            runner.logger.info(log_str)
            runner.visualizer.add_scalars(
                tag, step=runner.iter + 1, file_path=self.json_log_path)
            self.log_metrics(tag, step=runner.iter + 1)



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


def make_annotation(sample, img_id, annot_id, dict_json):
    dict_json['images'].append({'file_name': sample['filename'],
                                'height': sample['height'],
                                'width': sample['width'],
                                'id': img_id})
    for i, annot in enumerate(sample['annotations']):
        if 'bbox' in annot:
            instance = {}
            instance['id'] = annot_id + i + 1
            instance['iscrowd'] = 0
            instance['category_id'] = annot["category_id"]
            instance['ignore'] = False
            instance['image_id'] = img_id
            bbox = annot['bbox']
            x, y, w, h = bbox
            instance['bbox'] = bbox
            if "segmentation_poly" in annot:
                if len(annot["segmentation_poly"]):
                    instance["segmentation"] = annot["segmentation_poly"]
                else:
                    instance['segmentation'] = [[x, y, x + w, y, x + w, y + h, x, y + h]]
            else:
                instance['segmentation'] = [[x, y, x + w, y, x + w, y + h, x, y + h]]
            instance['area'] = sum([poly_area(poly) for poly in instance['segmentation']])
            dict_json['annotations'].append(instance)

    return annot_id + i + 1


def prepare_dataset(ikdata, dataset_path, split_ratio):
    train_file = os.path.join(dataset_path, 'instances_train.json')
    test_file = os.path.join(dataset_path, 'instances_test.json')

    os.makedirs(dataset_path, exist_ok=True)
    print("Preparing dataset...")

    images = ikdata['images']
    n = len(images)
    train_idx = random.sample(range(n), int(n * split_ratio))
    json_train = {'categories': [{"id": k, "name": v} for k, v in ikdata['metadata']['category_names'].items()],
                  'images': [],
                  'annotations': []}
    json_test = copy.deepcopy(json_train)
    annot_id = 0
    for id, sample in enumerate(images):
        if id in train_idx:
            annot_id = make_annotation(sample, id, annot_id, json_train)
        else:
            annot_id = make_annotation(sample, id, annot_id, json_test)

    with open(train_file, 'w') as f:
        json.dump(json_train, f)
    with open(test_file, 'w') as f:
        json.dump(json_test, f)

    print("Dataset prepared!")
