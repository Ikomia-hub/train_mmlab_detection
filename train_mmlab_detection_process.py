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
import copy
import os
import logging
from typing import Union, Dict
from datetime import datetime

from ikomia import core, dataprocess, utils
from ikomia.core.task import TaskParam
from ikomia.dnn import dnntrain
from ikomia.core import config as ikcfg

from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmengine.visualization import Visualizer
from mmdet.utils import register_all_modules

from train_mmlab_detection.utils import prepare_dataset, register_mmlab_modules, search_and_modify_cfg


logger = logging.getLogger()
ConfigType = Union[Dict, Config, ConfigDict]


class MyRunner(Runner):

    @classmethod
    def from_custom_cfg(cls, cfg: ConfigType, custom_hooks: ConfigType, visualizer) -> 'Runner':
        """Build a runner from config.

        Args:
            cfg (ConfigType): A config used for building runner. Keys of
                ``cfg`` can see :meth:`__init__`.

        Returns:
            Runner: A runner build from ``cfg``.
        """
        cfg = copy.deepcopy(cfg)
        runner = cls(
            model=cfg['model'],
            work_dir=cfg['work_dir'],
            train_dataloader=cfg.get('train_dataloader'),
            val_dataloader=cfg.get('val_dataloader'),
            test_dataloader=cfg.get('test_dataloader'),
            train_cfg=cfg.get('train_cfg'),
            val_cfg=cfg.get('val_cfg'),
            test_cfg=cfg.get('test_cfg'),
            auto_scale_lr=cfg.get('auto_scale_lr'),
            optim_wrapper=cfg.get('optim_wrapper'),
            param_scheduler=cfg.get('param_scheduler'),
            val_evaluator=cfg.get('val_evaluator'),
            test_evaluator=cfg.get('test_evaluator'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=custom_hooks,
            data_preprocessor=cfg.get('data_preprocessor'),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            launcher=cfg.get('launcher', 'none'),
            env_cfg=cfg.get('env_cfg'),  # type: ignore
            log_processor=cfg.get('log_processor'),
            log_level=cfg.get('log_level', 'INFO'),
            visualizer=visualizer,
            default_scope=cfg.get('default_scope', 'mmengine'),
            randomness=cfg.get('randomness', dict(seed=None)),
            experiment_name=cfg.get('experiment_name'),
            cfg=cfg,
        )
        return runner


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class TrainMmlabDetectionParam(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)
        # Place default value initialization here
        self.cfg["cfg"] = "configs/yolox/yolox_s_8xb8-300e_coco.py"
        self.cfg["model_name"] = "yolox"
        self.cfg["model_weight_file"] = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/" \
                         "yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"
        self.cfg["epochs"] = 10
        self.cfg["batch_size"] = 2
        self.cfg["dataset_split_ratio"] = 90
        self.cfg["output_folder"] = os.path.dirname(os.path.realpath(__file__)) + "/runs/"
        self.cfg["eval_period"] = 1
        plugin_folder = os.path.dirname(os.path.realpath(__file__))
        self.cfg["dataset_folder"] = os.path.join(plugin_folder, 'dataset')
        self.cfg["use_expert_mode"] = False
        self.cfg["config_file"] = ""

    def set_values(self, param_map):
        self.cfg["model_name"] = param_map["model_name"]
        self.cfg["model_weight_file"] = param_map["model_weight_file"]
        self.cfg["cfg"] = param_map["cfg"]
        self.cfg["epochs"] = int(param_map["epochs"])
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["dataset_split_ratio"] = int(param_map["dataset_split_ratio"])
        self.cfg["output_folder"] = param_map["output_folder"]
        self.cfg["eval_period"] = int(param_map["eval_period"])
        self.cfg["dataset_folder"] = param_map["dataset_folder"]
        self.cfg["use_expert_mode"] = utils.strtobool(param_map["use_expert_mode"])
        self.cfg["config_file"] = param_map["config_file"]


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class TrainMmlabDetection(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)

        # Create parameters class
        self.output_dir = None
        self.stop_train = False

        if param is None:
            self.set_param_object(TrainMmlabDetectionParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        param = self.get_param_object()
        return param.cfg["epochs"]

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()
        self.stop_train = False
        # Get input :
        ikdataset = self.get_input(0)
        param = self.get_param_object()
        str_datetime = datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss")
        tb_logdir = os.path.join(ikcfg.main_cfg["tensorboard"]["log_uri"], str_datetime)
        split = param.cfg["dataset_split_ratio"] / 100

        # Output directory
        self.output_dir = os.path.join(param.cfg["output_folder"], str_datetime)
        os.makedirs(self.output_dir, exist_ok=True)

        prepare_dataset(ikdataset.data, param.cfg["dataset_folder"], split)
        register_mmlab_modules()

        if param.cfg["use_expert_mode"]:
            config = param.cfg["config_file"]
            cfg = Config.fromfile(config)
        else:
            if os.path.isfile(param.cfg["config_file"]):
                config = param.cfg["config_file"]
            else:
                config = os.path.join(os.path.dirname(os.path.abspath(__file__)), param.cfg["cfg"])

            cfg = Config.fromfile(config)
            classes = list(ikdataset.data["metadata"]["category_names"].values())
            search_and_modify_cfg(cfg, "num_classes", len(classes))
            cfg.work_dir = self.output_dir
            eval_period = param.cfg["eval_period"]

            train_dataset = dict(
                # use MultiImageMixDataset wrapper to support mosaic and mixup
                type='MultiImageMixDataset',
                dataset=dict(
                    metainfo=dict(classes=classes),
                    type="CocoDataset",
                    ann_file=os.path.join(param.cfg["dataset_folder"], 'instances_train.json'),
                    data_prefix=dict(img=''),
                    pipeline=[
                        dict(type='LoadImageFromFile', backend_args=None),
                        dict(type='LoadAnnotations', with_bbox=True)
                    ],
                    filter_cfg=dict(filter_empty_gt=False, min_size=32),
                    backend_args=None),
                pipeline=cfg.train_pipeline)

            test_dataset = dict(
                # use MultiImageMixDataset wrapper to support mosaic and mixup
                type='MultiImageMixDataset',
                dataset=dict(
                    type="CocoDataset",
                    metainfo= dict(classes=classes),
                    ann_file=os.path.join(param.cfg["dataset_folder"], 'instances_test.json'),
                    data_prefix=dict(img=''),
                    pipeline=[
                        dict(type='LoadImageFromFile', backend_args=None),
                        dict(type='LoadAnnotations', with_bbox=True)
                    ],
                    filter_cfg=dict(filter_empty_gt=False, min_size=32),
                    backend_args=None),
                pipeline=cfg.test_pipeline)

            cfg.train_dataloader.dataset = train_dataset
            cfg.test_dataloader.dataset = test_dataset
            cfg.val_dataloader.dataset = test_dataset

            cfg.train_dataloader.batch_size = param.cfg["batch_size"]
            cfg.train_dataloader.num_workers = 0
            cfg.train_dataloader.persistent_workers = False

            cfg.test_dataloader.batch_size = param.cfg["batch_size"]
            cfg.test_dataloader.num_workers = 0
            cfg.test_dataloader.persistent_workers = False

            cfg.val_dataloader.batch_size = param.cfg["batch_size"]
            cfg.val_dataloader.num_workers = 0
            cfg.val_dataloader.persistent_workers = False

            cfg.load_from = param.cfg["model_weight_file"]

            cfg.train_cfg.max_epochs = param.cfg["epochs"]
            cfg.train_cfg.val_interval = eval_period

            cfg.val_evaluator = dict(
                type='CocoMetric',
                ann_file=os.path.join(param.cfg["dataset_folder"], 'instances_test.json'),
                metric='bbox', #'segm' for segmentation metric
                backend_args=None)
            cfg.test_evaluator = cfg.val_evaluator

        amp = True
        # save only best and last checkpoint
        cfg.checkpoint_config = None

        if "checkpoint" in cfg.default_hooks:
            cfg.default_hooks.checkpoint["interval"] = -1
            cfg.default_hooks.checkpoint["save_best"] = 'coco/bbox_mAP'
            cfg.default_hooks.checkpoint["rule"] = 'greater'

        cfg.visualizer.vis_backends = [dict(type='TensorboardVisBackend', save_dir=tb_logdir)]

        try:
            visualizer = Visualizer.get_current_instance()
        except:
            visualizer = cfg.get('visualizer')

        # register all modules in mmdet into the registries
        # do not init the default scope here because it will be init in the runner
        try:
            register_all_modules(init_default_scope=False)
        except:
            pass

        # enable automatic-mixed-precision training
        if amp:
            optim_wrapper = cfg.optim_wrapper.type
            if optim_wrapper == 'AmpOptimWrapper':
                print_log(
                    'AMP training is already enabled in your config.',
                    logger='current',
                    level=logging.WARNING)
            else:
                assert optim_wrapper == 'OptimWrapper', (
                    '`--amp` is only supported when the optimizer wrapper type is '
                    f'`OptimWrapper` but got {optim_wrapper}.')
                cfg.optim_wrapper.type = 'AmpOptimWrapper'
                cfg.optim_wrapper.loss_scale = 'dynamic'

        custom_hooks = [
            dict(type='CustomHook', stop=self.get_stop, output_folder=str(self.output_folder),
                 emit_step_progress=self.emit_step_progress, priority='LOWEST'),
            dict(type='CustomLoggerHook', log_metrics=self.log_metrics)
        ]

        # build the runner from config
        runner = MyRunner.from_custom_cfg(cfg, custom_hooks, visualizer)

        # add custom hook to stop process and save the latest model each epoch
        runner.cfg = cfg

        print("Start training")
        # start training
        runner.train()

        print("Training finished!")
        # Call end_task_run to finalize process
        self.end_task_run()

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
        self.info.short_description = "Train for MMLAB detection models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Detection"
        self.info.icon_path = "icons/mmlab.png"
        self.info.version = "1.2.0"
        self.info.max_python_version = "3.9"
        self.info.max_python_version = "3.11"
        self.info.min_ikomia_version = "0.15.0"
        # self.info.icon_path = "your path to a specific icon"
        self.info.authors = "Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and" \
                            "Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and" \
                            "Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and" \
                            "Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and" \
                            "Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong" \
                            "and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua"
        self.info.article = "{MMDetection}: Open MMLab Detection Toolbox and Benchmark"
        self.info.journal = "publication journal"
        self.info.year = 2019
        self.info.license = "Apache 2.0"
        # URL of documentation
        self.info.documentation_link = "https://mmdetection.readthedocs.io/en/latest/"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/train_mmlab_detection"
        self.info.original_repository = "https://github.com/open-mmlab/mmdetection"
        # Keywords used for search
        self.info.keywords = "train, mmlab, mmdet, detection"
        self.info.algo_type = core.AlgoType.TRAIN
        self.info.algo_tasks = "OBJECT_DETECTION"
        self.info.hardware_config.min_cpu = 4
        self.info.hardware_config.min_ram = 16
        self.info.hardware_config.gpu_required = True
        self.info.hardware_config.min_vram = 16

    def create(self, param=None):
        # Create process object
        return TrainMmlabDetection(self.info.name, param)
