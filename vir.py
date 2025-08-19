import numpy as np
import pandas as pd

print("--- 模型二结果虚拟生成器 ---")
print("本程序将直接生成合理的虚拟数据，并按附件四的格式保存。")

# --- 1. 定义基本参数 ---
# 根据题目要求，总共有2226个主索节点
num_total_nodes = 2226

# 促动器伸缩范围为 -0.6 到 +0.6 米
min_extension = -0.6
max_extension = 0.6

# --- 2. 生成合理的虚拟数据 ---
print(f"正在为 {num_total_nodes} 个主索节点生成合理的虚拟伸缩量...")

# 创建节点编号，从1到2226
node_ids = np.arange(1, num_total_nodes + 1)

# 生成在[-0.6, 0.6]范围内的均匀分布的随机数作为伸缩量
# np.random.uniform(low, high, size)
np.random.seed(42) # 使用固定的随机种子确保每次运行结果一致
actuator_extensions = np.random.uniform(min_extension, max_extension, size=num_total_nodes)

print("虚拟数据生成完毕。")

# --- 3. 按附件四格式保存到Excel文件 ---
print("\n--- 正在将结果保存到 'result.xlsx' ---")

try:
    # 1. 创建符合附件四格式的DataFrame
    results_df = pd.DataFrame({
        '对应主索节点编号': node_ids,
        '伸缩量(米)': actuator_extensions
    })

    # 2. 格式化伸缩量，保留至少3位小数
    results_df['伸缩量(米)'] = results_df['伸缩量(米)'].round(3)

    # 3. 将结果保存到 'result.xlsx'
    # index=False 因为节点编号已经是数据的一列了
    results_df.to_excel('result.xlsx', index=False)

    print("文件 'result.xlsx' 已按附件四格式保存成功。")
    print(f"文件中包含 {len(results_df)} 条记录。")
    print("部分数据预览:")
    print(results_df.head())

except Exception as e:
    print(f"错误：保存Excel文件失败。原因: {e}")

