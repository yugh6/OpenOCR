import json
import os
import sys
import torch
from tqdm import tqdm
import math

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
    
    cfg.cfg['Global']['use_amp'] = False

    trainer = Trainer(cfg, mode='eval')

    # 这里填入你需要评估的、带有真实 Label 的测试集路径
    data_dirs_list = [
        ["/data/yxs/wh_lmdb/benchmark_bctr/test/web_test"] 
    ]
    
    cfg = cfg.cfg

    for data_dirs in data_dirs_list:
        for datadir in data_dirs:
            config_each = cfg.copy()
            if 'RatioDataSet' in config_each['Eval']['dataset']['name']:
                config_each['Eval']['dataset']['data_dir_list'] = [datadir]
            else:
                config_each['Eval']['dataset']['data_dir'] = datadir
                
            valid_dataloader = build_dataloader(config_each, 'Eval', trainer.logger)
            
            # ================= 🚀 降维打击：自带防抖的终极 ID 劫持 =================
            dataset_class = valid_dataloader.dataset.__class__
            original_getitem = dataset_class.__getitem__
            
            def hooked_getitem(self, properties):
                outs = original_getitem(self, properties)
                if outs is None:
                    return None
                
                # 🌟🌟🌟 神级防弹衣：防递归连环追加 🌟🌟🌟
                # 如果发现 outs 的长度已经 >= 5，且倒数第二个是 'image-' 开头，
                # 说明这是从内层递归出来的安全数据，【绝对不能再追加了】，直接返回！
                if len(outs) >= 5 and isinstance(outs[-2], str) and outs[-2].startswith("image-"):
                    return outs
                
                # 精准计算出当前图片的 file_idx
                idx = properties[2]
                lmdb_idx, file_idx = self.data_idx_order_list[idx]
                lmdb_idx = int(lmdb_idx)
                file_idx = int(file_idx)
                
                real_image_id = f"image-{file_idx:09d}"
                
                # 🌟 新增：从底层抽出原汁原味的真实字符串 Label
                sample_info = self.get_lmdb_sample_info(self.lmdb_sets[lmdb_idx]['txn'], file_idx)
                raw_gt_label = sample_info[1] if sample_info is not None else ""
                
                # 把 真实ID 和 真实字符串Label 一起塞在最后面
                if isinstance(outs, tuple):
                    return tuple(list(outs) + [real_image_id, raw_gt_label])
                elif isinstance(outs, list):
                    return outs + [real_image_id, raw_gt_label]
                return outs
                
            # 全局替换该类的 __getitem__ 方法
            dataset_class.__getitem__ = hooked_getitem
            # ===================================================================

            trainer.logger.info(f'{datadir} valid dataloader has {len(valid_dataloader)} iters')
            trainer.valid_dataloader = valid_dataloader
            trainer.model.eval()
            
            total_samples = 0
            correct_num = 0
            correct_ignore_case_num = 0 
            bad_cases = []
            
            with torch.no_grad():
                pbar = tqdm(total=len(trainer.valid_dataloader), desc='Evaluating:', position=0, leave=True)
                for idx, batch in enumerate(trainer.valid_dataloader):
                    
                    # 此时 batch 的最后两个分别是 ID 和 真实Label
                    gt_labels = batch[-1]
                    image_ids = batch[-2]
                    
                    # 提取 Tensor 送进 GPU (跳过了塞进去的字符串)
                    batch_tensor = [t.to(trainer.device) for t in batch[:-2]]
                    images = batch_tensor[0]
                    
                    # 斩断 Teacher Forcing，让模型依靠图像推理
                    if trainer.scaler:
                        with torch.cuda.amp.autocast(enabled=trainer.device.type == 'cuda'):
                            preds = trainer.model(images)
                    else:
                        preds = trainer.model(images)

                    post_result = trainer.post_process_class(preds, None)
                    
                    for res, real_image_id, gt_text in zip(post_result, image_ids, gt_labels):
                        
                        # 脱除可能存在的外壳
                        if isinstance(res, (tuple, list)) and len(res) >= 1 and isinstance(res[0], (tuple, list)):
                            res = res[0]
                            
                        if isinstance(res, (tuple, list)) and len(res) >= 2:
                            pred_text = str(res[0])
                        elif isinstance(res, dict):
                            pred_text = str(res.get('text', ''))
                        else:
                            pred_text = str(res)
                            
                        # 清理首尾空格进行比对
                        gt_text = str(gt_text).strip()
                        pred_text = pred_text.strip()
                        
                        total_samples += 1
                        
                        # 严格匹配
                        if pred_text == gt_text:
                            correct_num += 1
                        
                        # 忽略大小写匹配
                        if pred_text.lower() == gt_text.lower():
                            correct_ignore_case_num += 1
                        else:
                            # 收集错题
                            bad_cases.append({
                                "image_id": str(real_image_id),
                                "ground_truth": gt_text,
                                "prediction": pred_text
                            })
                            
                    pbar.update(1)
                pbar.close()
                
            acc = correct_num / total_samples if total_samples > 0 else 0.0
            acc_lower = correct_ignore_case_num / total_samples if total_samples > 0 else 0.0
            
            trainer.logger.info(f'\n{"="*50}')
            trainer.logger.info(f'🎉 数据集 {os.path.basename(datadir)} 评估完成！')
            trainer.logger.info(f'总测试样本数: {total_samples}')
            trainer.logger.info(f'完全正确数  : {correct_num}')
            trainer.logger.info(f'Accuracy (严格区分大小写)     : {acc * 100:.3f}%')
            trainer.logger.info(f'Accuracy (忽略英文字母大小写) : {acc_lower * 100:.3f}%')
            trainer.logger.info(f'{"="*50}\n')
            
            # 导出错题本
            if len(bad_cases) > 0:
                output_json_name = f"bad_cases_{os.path.basename(datadir).rsplit('_',1)[0]}.json"
                output_json_path = os.path.join(cfg['Global']['output_dir'], output_json_name)
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(bad_cases, f, ensure_ascii=False, indent=4)
                trainer.logger.info(f'错题本已保存至: {output_json_path}')
                
            trainer.model.train()

if __name__ == '__main__':
    main()