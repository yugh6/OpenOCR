# import json
# import os
# import matplotlib.pyplot as plt
# import numpy as np

# if not os.path.exists("/remote-home/yexingsun/OpenOCR/output/score_distribution"):
#     os.makedirs("/remote-home/yexingsun/OpenOCR/output/score_distribution")
# os.chdir("/remote-home/yexingsun/OpenOCR/output/score_distribution")

# lmdb_name_list = [
#                     "book32_lmdb",
#                     "cc_lmdb",
#                     "openvino_lmdb",
#                     "hand",
#                     "subdir1",
#                     "subdir2"
#                 ] # 你可以同时分析多个数据集的分数分布
# for lmdb_name in lmdb_name_list:
#     json_path = f"/remote-home/yexingsun/OpenOCR/output/json/pseudo_labels_{lmdb_name}.json"

#     print(f"正在读取文件: {os.path.basename(json_path)} ...")
#     scores = []

#     with open(json_path, 'r', encoding='utf-8') as f:
#         try:
#             data = json.load(f)
#             scores = [item['score'] for item in data]
#         except json.JSONDecodeError:
#             f.seek(0)
#             for line in f:
#                 if line.strip():
#                     scores.append(json.loads(line)['score'])

#     total_samples = len(scores)
#     print(f"✅ 成功加载 {total_samples} 条数据的置信度分数。\n")

#     # ================= 打印统计报表 =================

#     scores_arr = np.array(scores, dtype=float)
#     nan_count = np.isnan(scores_arr).sum()
#     if nan_count > 0:
#         # 将所有的 nan 替换为 0.0
#         scores_arr = np.nan_to_num(scores_arr, nan=0.0)
#         scores = scores_arr.tolist()

#     print("-" * 40)
#     print(f"📊 {lmdb_name} 置信度分数统计报表 📊")
#     print("-" * 40)
#     print(f"平均分 (Mean) : {np.mean(scores_arr):.4f}")
#     print(f"最高分 (Max)  : {np.max(scores_arr):.4f}")
#     print(f"最低分 (Min)  : {np.min(scores_arr):.4f}")
#     print("-" * 40)

#     # 统计不同阈值下的数据量
#     thresholds_to_check = [0.80, 0.85, 0.90, 0.95, 0.98]
#     for t in thresholds_to_check:
#         keep_count = np.sum(scores_arr >= t)
#         keep_ratio = keep_count / total_samples * 100
#         print(f"阈值 >= {t:.2f} | 可保留数据量: {keep_count:>7d} 张 ({keep_ratio:>5.2f}%)")
#     print("-" * 40)


#     # ================= 绘制分布直方图 =================
#     plt.figure(figsize=(10, 6), dpi=150)

#     # 画直方图 (将 0~1 分成 50 个区间)
#     n, bins, patches = plt.hist(scores, bins=50, range=(0, 1), color='#4CAF50', edgecolor='black', alpha=0.8)

#     # 添加标题和标签
#     plt.title(f'{lmdb_name} - Distribution of Pseudo-Label Confidence Scores', fontsize=15, fontweight='bold')
#     plt.xlabel('Confidence Score', fontsize=12)
#     plt.ylabel('Number of Images', fontsize=12)

#     # 添加网格线，方便看数量
#     plt.grid(axis='y', linestyle='--', alpha=0.7)

#     # 标出我们最关心的 0.90 阈值线
#     plt.axvline(x=0.90, color='red', linestyle='dashed', linewidth=2, label='Threshold 0.90')
#     plt.legend()

#     # 保存图片
#     output_img = f'score_distribution_{lmdb_name}.png'
#     plt.savefig(output_img, bbox_inches='tight')
#     print(f"\n📈 分数分布直方图已保存至当前目录: {output_img}\n")
    
#     # 关闭当前图表，释放内存，防止下一个数据集画图时叠加上去
#     plt.close()


import lmdb
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

# 配置输出目录
output_dir = "/remote-home/yexingsun/OpenOCR/output/score_distribution_1"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# 注意：通常不建议在脚本中随意 chdir，这里为了保持原逻辑保留，但建议后续使用绝对路径保存
os.chdir(output_dir)

# LMDB 根目录 (假设所有 lmdb 都在这个目录下，或者你需要提供完整路径)
# 如果 lmdb_name 只是文件夹名，请确保 base_lmdb_root 设置正确
base_lmdb_root = "/data/yxs/ygh/labeled/UCTI-11M" 

lmdb_name_list = [
    # "book32_lmdb",
    # "cc_lmdb",
    # "openvino_lmdb",
    "hand",
    # "subdir1",
    # "subdir2"
]

def extract_scores_from_lmdb(lmdb_path):
    """
    从 LMDB 中提取所有样本的 score
    """
    scores = []
    print(f"正在打开 LMDB: {lmdb_path}")
    
    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    except Exception as e:
        print(f"❌ 无法打开 LMDB: {e}")
        return []

    with env.begin() as txn:
        # 获取总样本数
        num_samples_buf = txn.get(b"num-samples")
        if not num_samples_buf:
            print("❌ 错误: 找不到 'num-samples' 键。")
            env.close()
            return []
        
        num_samples = int(num_samples_buf.decode())
        print(f"数据集总样本数: {num_samples}")
        print("开始提取 Score...")

        for i in range(1, num_samples + 1):
            # 构建 key: score-000000001
            score_key = f"score-{i:09d}".encode()
            score_buf = txn.get(score_key)
            
            try:
                # 尝试解码为 float
                # 假设 score 存储的是字符串形式的数字，如 "0.95"
                score_val = float(score_buf.decode('utf-8').strip())
                scores.append(score_val)
            except (ValueError, UnicodeDecodeError) as e:
                # 如果解码失败或转换失败，跳过或记为无效
                print(f"警告: 样本 {i} score 格式错误: {e}")
                pass
            
    env.close()
    return scores

for lmdb_name in lmdb_name_list:
    # 构建完整的 LMDB 路径
    # 注意：请根据实际路径结构调整这里，有时 lmdb 路径就是 base/lmdb_name，有时可能是 base/lmdb_name/data.mdb 的父目录
    lmdb_path = os.path.join(base_lmdb_root, lmdb_name)
    
    # 检查路径是否存在
    if not os.path.exists(lmdb_path):
        print(f"⚠️ 路径不存在，跳过: {lmdb_path}")
        continue

    print(f"\n{'='*20} 处理数据集: {lmdb_name} {'='*20}")
    
    # 1. 提取分数
    scores = extract_scores_from_lmdb(lmdb_path)
    
    total_samples = len(scores)
    if total_samples == 0:
        print(f"⚠️ {lmdb_name} 未提取到任何有效分数，跳过绘图。\n")
        continue

    print(f"✅ 成功加载 {total_samples} 条数据的置信度分数。\n")

    # ================= 打印统计报表 =================
    scores_arr = np.array(scores, dtype=float)
    
    # 处理可能的 NaN (虽然上面提取时尽量避免了，但以防万一)
    nan_count = np.isnan(scores_arr).sum()
    if nan_count > 0:
        print(f"⚠️ 发现 {nan_count} 个 NaN 值，已替换为 0.0")
        scores_arr = np.nan_to_num(scores_arr, nan=0.0)

    print("-" * 40)
    print(f"📊 {lmdb_name} 置信度分数统计报表 📊")
    print("-" * 40)
    print(f"平均分 (Mean) : {np.mean(scores_arr):.4f}")
    print(f"最高分 (Max)  : {np.max(scores_arr):.4f}")
    print(f"最低分 (Min)  : {np.min(scores_arr):.4f}")
    print("-" * 40)

    # 统计不同阈值下的数据量
    thresholds_to_check = [0.80, 0.85, 0.90, 0.95, 0.98]
    for t in thresholds_to_check:
        keep_count = np.sum(scores_arr >= t)
        keep_ratio = keep_count / total_samples * 100
        print(f"阈值 >= {t:.2f} | 可保留数据量: {keep_count:>7d} 张 ({keep_ratio:>5.2f}%)")
    print("-" * 40)

    # ================= 绘制分布直方图 =================
    plt.figure(figsize=(10, 6), dpi=150)

    # 画直方图 (将 0~1 分成 50 个区间)
    # 注意：如果分数集中在 1.0 附近，可能需要调整 bins 或 range
    n, bins, patches = plt.hist(scores_arr, bins=50, range=(0, 1), color='#4CAF50', edgecolor='black', alpha=0.8)

    # 添加标题和标签
    plt.title(f'{lmdb_name} - Distribution of Confidence Scores', fontsize=15, fontweight='bold')
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)

    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 标出我们最关心的 0.90 阈值线
    plt.axvline(x=0.90, color='red', linestyle='dashed', linewidth=2, label='Threshold 0.90')
    plt.legend()

    # 保存图片
    output_img = f'score_distribution_{lmdb_name}.png'
    plt.savefig(output_img, bbox_inches='tight')
    print(f"\n📈 分数分布直方图已保存至: {os.path.join(output_dir, output_img)}\n")
    
    # 关闭当前图表
    plt.close()