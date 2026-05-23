import json
import os
import matplotlib.pyplot as plt
import numpy as np

if not os.path.exists("/remote-home/yexingsun/OpenOCR/output/score_distribution"):
    os.makedirs("/remote-home/yexingsun/OpenOCR/output/score_distribution")
os.chdir("/remote-home/yexingsun/OpenOCR/output/score_distribution")

lmdb_name_list = [
                    "book32_lmdb",
                    "cc_lmdb",
                    "openvino_lmdb",
                    "hand",
                    "subdir1",
                    "subdir2"
                ] # 你可以同时分析多个数据集的分数分布

for lmdb_name in lmdb_name_list:
    json_path = f"/remote-home/yexingsun/OpenOCR/output/json/pseudo_labels_{lmdb_name}.json"
    
    # 防止文件不存在导致报错中断
    if not os.path.exists(json_path):
        print(f"⚠️ 警告: 找不到文件 {json_path}，跳过该数据集。\n")
        continue

    print(f"正在读取文件: {os.path.basename(json_path)} ...")
    scores = []

    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            scores = [item['score'] for item in data]
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                if line.strip():
                    scores.append(json.loads(line)['score'])

    total_samples = len(scores)
    if total_samples == 0:
        print(f"⚠️ {lmdb_name} 数据为空，跳过。\n")
        continue
        
    print(f"✅ 成功加载 {total_samples} 条数据的置信度分数。\n")

    # ================= 打印统计报表 =================

    scores_arr = np.array(scores, dtype=float)
    nan_count = np.isnan(scores_arr).sum()
    if nan_count > 0:
        # 将所有的 nan 替换为 0.0
        scores_arr = np.nan_to_num(scores_arr, nan=0.0)
        scores = scores_arr.tolist()

    print("-" * 45)
    print(f"📊 {lmdb_name} 置信度分数统计报表 📊")
    print("-" * 45)
    print(f"平均分 (Mean) : {np.mean(scores_arr):.4f}")
    print(f"最高分 (Max)  : {np.max(scores_arr):.4f}")
    print(f"最低分 (Min)  : {np.min(scores_arr):.4f}")
    print("-" * 45)

    # 统计不同阈值下的数据量
    thresholds_to_check = [0.80, 0.85, 0.90, 0.95, 0.98]
    for t in thresholds_to_check:
        keep_count = np.sum(scores_arr >= t)
        keep_ratio = keep_count / total_samples * 100
        print(f"阈值 >= {t:.2f} | 可保留数据量: {keep_count:>7d} 张 ({keep_ratio:>5.2f}%)")
    print("-" * 45)

    # ================= 新增：0.90-1.00 细化区间统计 =================
    print(f"🎯 [0.90, 1.00] 区间细化分布 (步长 0.01):")
    for i in range(10):
        # 避免浮点数精度问题，采用 round 限制为两位小数
        lower = round(0.90 + i * 0.01, 2)
        upper = round(lower + 0.01, 2)
        
        # 对于前9个区间，采用左闭右开 [lower, upper)
        if i < 9:
            mask = (scores_arr >= lower) & (scores_arr < upper)
            range_label = f"[{lower:.2f}, {upper:.2f})"
        # 最后一个区间 [0.99, 1.00]，采用双闭合，把刚好得满分 1.0 的也算进去
        else:
            mask = (scores_arr >= lower) & (scores_arr <= upper)
            range_label = f"[{lower:.2f}, {upper:.2f}]"
            
        interval_count = np.sum(mask)
        interval_ratio = (interval_count / total_samples) * 100
        
        print(f"  区间 {range_label} : {interval_count:>7d} 张 ({interval_ratio:>5.2f}%)")
    print("-" * 45)


    # ================= 绘制分布直方图 =================
    plt.figure(figsize=(10, 6), dpi=150)

    # 画直方图 (将 0~1 分成 50 个区间)
    n, bins, patches = plt.hist(scores, bins=50, range=(0, 1), color='#4CAF50', edgecolor='black', alpha=0.8)

    # 添加标题和标签
    plt.title(f'{lmdb_name} - Distribution of Pseudo-Label Confidence Scores', fontsize=15, fontweight='bold')
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)

    # 添加网格线，方便看数量
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 标出我们最关心的 0.90 阈值线
    plt.axvline(x=0.90, color='red', linestyle='dashed', linewidth=2, label='Threshold 0.90')
    plt.legend()

    # 保存图片
    output_img = f'score_distribution_{lmdb_name}.png'
    plt.savefig(output_img, bbox_inches='tight')
    print(f"\n📈 分数分布直方图已保存至当前目录: {output_img}\n\n")
    
    # 关闭当前图表，释放内存，防止下一个数据集画图时叠加上去
    plt.close()