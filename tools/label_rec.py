import csv
import os
import sys
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.data import build_dataloader
from tools.engine.config import Config
from tools.engine.trainer import Trainer
from tools.utility import ArgsParser


def parse_args():
    parser = ArgsParser()
    args = parser.parse_args()
    return args


def main():
    FLAGS = parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)

    msr = False
    if 'RatioDataSet' in cfg.cfg['Eval']['dataset']['name']:
        msr = True

    if cfg.cfg['Global']['output_dir'][-1] == '/':
        cfg.cfg['Global']['output_dir'] = cfg.cfg['Global']['output_dir'][:-1]
    if cfg.cfg['Global']['pretrained_model'] is None:
        cfg.cfg['Global'][
            'pretrained_model'] = cfg.cfg['Global']['output_dir'] + '/best.pth'
    cfg.cfg['Global']['use_amp'] = False
    cfg.cfg['PostProcess']['with_ratio'] = True
    cfg.cfg['Metric']['with_ratio'] = True
    cfg.cfg['Metric']['max_len'] = 25
    cfg.cfg['Metric']['max_ratio'] = 12
    # cfg.cfg['Eval']['dataset']['transforms'][-1]['KeepKeys'][
        # 'keep_keys'].append('real_ratio')
    trainer = Trainer(cfg, mode='eval')

    best_model_dict = trainer.status.get('metrics', {})
    trainer.logger.info('metric in ckpt ***************')
    for k, v in best_model_dict.items():
        trainer.logger.info('{}:{}'.format(k, v))

    data_dirs_list = [
        [
                "/data/yxs/ygh/unlabeled/chinese/subdir2_std",
        ],
    ]
    fixed_path = "/data/yxs/ygh/labeled/UCTI-11M"
    cfg = cfg.cfg
    # cfg['Eval']['dataset']['name'] = cfg['Eval']['dataset']['name'] + 'Label'
    cfg['Eval']['dataset']['transforms'][-1]['KeepKeys']['keep_keys'] = ['image', 'label', 'length', 'ori_image']
    for data_dirs in data_dirs_list:

        for datadir in data_dirs:
            config_each = cfg.copy()
            if msr:
                config_each['Eval']['dataset']['data_dir_list'] = [datadir]
            else:
                config_each['Eval']['dataset']['data_dir'] = datadir
            valid_dataloader = build_dataloader(config_each, 'Eval',
                                                trainer.logger)
            trainer.logger.info(
                f'{datadir} valid dataloader has {len(valid_dataloader)} iters'
            )
            trainer.valid_dataloader = valid_dataloader
            output_path = os.path.join(fixed_path, os.path.basename(os.path.normpath(datadir)).rsplit('_',1)[0])
            os.makedirs(output_path, exist_ok=True)
            trainer.label(output_path)

if __name__ == '__main__':
    main()
