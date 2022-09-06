# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
from ikomia.core.task import TaskParam
from ikomia.dnn import dnntrain
from ikomia.core import config as ikcfg

import copy
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist

from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_root_logger)
import mmcv
import os
import torch
from train_mmlab_detection.utils import prepare_dataset, UserStop, register_mmlab_modules, search_and_modify_cfg
from datetime import datetime
from distutils.util import strtobool
import logging

logger = logging.getLogger()


# Your imports below


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class TrainMmlabDetectionParam(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)
        # Place default value initialization here
        self.cfg["model_config"] = "yolox_tiny_8x8_300e_coco"
        self.cfg["model_name"] = "yolox"
        self.cfg["model_url"] = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco" \
                                "/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth "
        self.cfg["epochs"] = 10
        self.cfg["batch_size"] = 2
        self.cfg["dataset_split_percentage"] = 90
        self.cfg["output_folder"] = os.path.dirname(os.path.realpath(__file__)) + "/runs/"
        self.cfg["eval_period"] = 1
        plugin_folder = os.path.dirname(os.path.realpath(__file__))
        self.cfg["dataset_folder"] = os.path.join(plugin_folder, 'dataset')
        self.cfg["expert_mode"] = False
        self.cfg["custom_config"] = ""

    def setParamMap(self, param_map):
        self.cfg["model_name"] = param_map["model_name"]
        self.cfg["model_url"] = param_map["model_url"]
        self.cfg["model_config"] = param_map["model_config"]
        self.cfg["epochs"] = int(param_map["epochs"])
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["dataset_split_percentage"] = int(param_map["dataset_split_percentage"])
        self.cfg["output_folder"] = param_map["output_folder"]
        self.cfg["eval_period"] = int(param_map["eval_period"])
        self.cfg["dataset_folder"] = param_map["dataset_folder"]
        self.cfg["expert_mode"] = strtobool(param_map["expert_mode"])
        self.cfg["custom_config"] = param_map["custom_config"]


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class TrainMmlabDetection(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)

        # Create parameters class
        self.stop_train = False
        if param is None:
            self.setParam(TrainMmlabDetectionParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        param = self.getParam()
        return param.cfg["epochs"]

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()
        self.stop_train = False
        # Examples :
        # Get input :
        ikdataset = self.getInput(0)

        plugin_folder = os.path.dirname(os.path.abspath(__file__))

        param = self.getParam()

        str_datetime = datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss")

        save_dir = os.path.join(plugin_folder, "runs", str_datetime)

        tb_logdir = os.path.join(ikcfg.main_cfg["tensorboard"]["log_uri"], str_datetime)

        split = param.cfg["dataset_split_percentage"] / 100

        prepare_dataset(ikdataset.data, plugin_folder, split)
        register_mmlab_modules()

        config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", param.cfg["model_name"],
                              param.cfg["model_config"] + '.py')
        cfg = Config.fromfile(config)
        classes = list(ikdataset.data["metadata"]["category_names"].values())
        search_and_modify_cfg(cfg, "num_classes", len(classes))
        cfg.work_dir = save_dir
        eval_period = param.cfg["eval_period"]
        cfg.load_from = param.cfg["model_url"]
        cfg.log_config = dict(
            interval=5,

            hooks=[
                dict(type='TextLoggerHook'),
                dict(type='TensorboardLoggerHook',
                     log_dir=tb_logdir)
            ])
        cfg.total_epochs = cfg.max_epochs = cfg.runner.max_epochs = param.cfg["epochs"]

        cfg.evaluation = dict(interval=eval_period, metric="bbox", save_best="auto",
                              rule="greater")
        cfg.dataset_type = "CocoDataset"
        cfg.data_root = None
        cfg.device = 'cuda'
        cfg.train_dataset = dict(
            type='MultiImageMixDataset',
            dataset=dict(
                type=cfg.dataset_type,
                classes=classes,
                ann_file=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'dataset', 'instances_train.json'),
                img_prefix=None,
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True)
                ],
                filter_empty_gt=False,
            ),
            pipeline=cfg.train_pipeline)

        cfg.data = dict(
            samples_per_gpu=param.cfg["batch_size"],
            workers_per_gpu=0,
            train=cfg.train_dataset,
            test=dict(
                type=cfg.dataset_type,
                classes=classes,
                ann_file=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'dataset', 'instances_test.json'),
                img_prefix=None,
                pipeline=cfg.test_pipeline),
            val=dict(
                type=cfg.dataset_type,
                classes=classes,
                ann_file=os.path.join(os.path.abspath(os.path.dirname(__file__)), 'dataset', 'instances_test.json'),
                img_prefix=None,
                pipeline=cfg.test_pipeline))

        cfg.data.test_dataloader = dict(samples_per_gpu=1)

        gpus = 1
        launcher = "none"
        seed = None
        deterministic = True
        no_validate = cfg.evaluation.interval <= 0
        cfg.checkpoint_config = None
        # scale lr
        cfg.optimizer.lr = cfg.optimizer.lr / 64 * param.cfg["batch_size"]

        # import modules from string list.
        if cfg.get('custom_imports', None):
            from mmcv.utils import import_modules_from_strings

            import_modules_from_strings(**cfg['custom_imports'])
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        cfg.gpu_ids = range(1) if gpus is None else range(gpus)
        # init distributed env first, since logger depends on the dist info.
        if launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(launcher, **cfg.dist_params)
            # re-set gpu_ids with distributed training mode
            _, world_size = get_dist_info()
            cfg.gpu_ids = range(world_size)

        # create work_dir
        mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
        # dump config
        cfg.dump(os.path.join(cfg.work_dir, os.path.basename(config)))

        # init the logger before other steps
        timestamp = str_datetime

        # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
        meta = dict()
        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        meta['env_info'] = env_info
        meta['config'] = cfg.pretty_text
        # log some basic info
        logger.info(f'Distributed training: {distributed}')
        logger.info(f'Config:\n{cfg.pretty_text}')

        # set random seeds
        if seed is not None:
            logger.info(f'Set random seed to {seed}, '
                        f'deterministic: {deterministic}')
            set_random_seed(seed, deterministic=deterministic)
        cfg.seed = seed
        meta['seed'] = seed
        meta['exp_name'] = os.path.basename(config)

        datasets = [build_dataset(cfg.data.train)]

        cfg.custom_hooks = [
            dict(type='CustomHook', stop=self.get_stop, output_folder=cfg.work_dir,
                 emitStepProgress=self.emitStepProgress, priority='LOWEST'),
            dict(type='CustomMlflowLoggerHook', log_metrics=self.log_metrics)
        ]

        model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        model.init_weights()

        # add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES
        try:
            train_detector(
                model,
                datasets,
                cfg,
                distributed=distributed,
                validate=(not no_validate),
                timestamp=timestamp,
                meta=meta)
        except UserStop:
            logger.info("Training stopped by user")

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()

    def get_stop(self):
        return self.stop_train

    def stop(self):
        super().stop()
        self.stop_train = True


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class TrainMmlabDetectionFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "train_mmlab_detection"
        self.info.shortDescription = "your short description"
        self.info.description = "your description"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python"
        self.info.version = "1.0.0"
        # self.info.iconPath = "your path to a specific icon"
        self.info.authors = "algorithm author"
        self.info.article = "title of associated research article"
        self.info.journal = "publication journal"
        self.info.year = 2021
        self.info.license = "MIT License"
        # URL of documentation
        self.info.documentationLink = ""
        # Code source repository
        self.info.repository = ""
        # Keywords used for search
        self.info.keywords = "your,keywords,here"

    def create(self, param=None):
        # Create process object
        return TrainMmlabDetection(self.info.name, param)
