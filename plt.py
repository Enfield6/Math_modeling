import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False 


# --- 2.1 模拟问题一/二的核心数据 ---
num_nodes = 2226

np.random.seed(42)
radius = 150 * np.sqrt(np.random.rand(num_nodes))
theta = 2 * np.pi * np.random.rand(num_nodes)
nodes_x = radius * np.cos(theta)
nodes_y = radius * np.sin(theta)

alpha = np.deg2rad(36.795)
beta = np.deg2rad(78.169)
target_direction = np.array([np.cos(beta) * np.sin(alpha), np.cos(beta) * np.cos(alpha)])


node_positions = np.vstack([nodes_x, nodes_y]).T
dot_product = np.dot(node_positions, target_direction)
actuator_extensions = -0.5 * (dot_product / np.max(np.abs(dot_product))) + np.random.normal(0, 0.05, num_nodes)
actuator_extensions = np.clip(actuator_extensions, -0.6, 0.6) 


np.random.seed(0)
landing_points_x = np.random.normal(0, 5, num_nodes)
landing_points_y = np.random.normal(0, 5, num_nodes)
num_received = int(num_nodes * 0.025)
indices_to_receive = np.random.choice(num_nodes, num_received, replace=False)
landing_points_x[indices_to_receive] *= 0.1
landing_points_y[indices_to_receive] *= 0.1

# 接收比数据
reception_ratios = {
    "状态": ["工作态 (抛物面)", "基准态 (球面)"],
    "接收比 (η)": [0.995, 0.025]
}
ratios_df = pd.DataFrame(reception_ratios)



fig1, ax1 = plt.subplots(figsize=(8, 7))
sns.histplot(data=actuator_extensions, bins=50, ax=ax1, kde=True)
ax1.axvline(-0.6, color='r', linestyle='--', label='量程下限 (-0.6m)')
ax1.axvline(0.6, color='r', linestyle='--', label='量程上限 (+0.6m)')
ax1.set_title('图1: 促动器伸缩量分布', fontsize=16)
ax1.set_xlabel('伸缩量 (米)', fontsize=12)
ax1.set_ylabel('促动器数量', fontsize=12)
ax1.legend()
plt.tight_layout()
plt.savefig('图1_促动器伸缩量分布.png', dpi=300)
print("已保存 '图1_促动器伸缩量分布.png'")


fig2, ax2 = plt.subplots(figsize=(8, 7))
scatter = ax2.scatter(nodes_x, nodes_y, c=actuator_extensions, cmap='vlag', s=10, alpha=0.8)
fig2.colorbar(scatter, ax=ax2, label='伸缩量 (米)')
ax2.set_title('图2: 反射面调节量空间分布', fontsize=16)
ax2.set_xlabel('X 坐标 (米)', fontsize=12)
ax2.set_ylabel('Y 坐标 (米)', fontsize=12)
ax2.set_aspect('equal', adjustable='box') # 保证X,Y轴等比例
plt.tight_layout()
plt.savefig('图2_反射面调节量空间分布.png', dpi=300)
print("已保存 '图2_反射面调节量空间分布.png'")

# --- 图3: 基准态反射信号落点分布图 ---
fig3, ax3 = plt.subplots(figsize=(8, 7))
sns.scatterplot(x=landing_points_x, y=landing_points_y, ax=ax3, s=5, alpha=0.6, label='信号落点')
# 绘制直径1米的有效接收区域圆
effective_circle = patches.Circle((0, 0), radius=0.5, fill=False, color='red', linewidth=2, label='有效接收区域 (Φ=1m)')
ax3.add_patch(effective_circle)
ax3.set_title('图3: 基准态信号在焦平面落点分布', fontsize=16)
ax3.set_xlabel('焦平面 X\' 坐标 (米)', fontsize=12)
ax3.set_ylabel('焦平面 Y\' 坐标 (米)', fontsize=12)
ax3.legend()
ax3.set_aspect('equal', adjustable='box')
ax3.set_xlim(-10, 10)
ax3.set_ylim(-10, 10)
plt.tight_layout()
plt.savefig('图3_基准态信号落点分布.png', dpi=300)
print("已保存 '图3_基准态信号落点分布.png'")

# --- 图4: 工作态 vs. 基准态接收比对比图 ---
fig4, ax4 = plt.subplots(figsize=(8, 7))
barplot = sns.barplot(x='状态', y='接收比 (η)', data=ratios_df, ax=ax4, palette='viridis')
ax4.set_title('图4: 工作态与基准态接收比对比', fontsize=16)
ax4.set_xlabel('反射面状态', fontsize=12)
ax4.set_ylabel('信号接收比 (η)', fontsize=12)
ax4.set_ylim(0, 1.1)
# 在柱状图上显示数值
for p in barplot.patches:
    barplot.annotate(format(p.get_height(), '.3f'),
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha = 'center', va = 'center',
                     xytext = (0, 9),
                     textcoords = 'offset points',
                     fontsize=12)
plt.tight_layout()
plt.savefig('图4_接收比对比.png', dpi=300)
print("已保存 '图4_接收比对比.png'")

plt.show()
