import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

# --- 1. 全局绘图设置 ---
# 设置支持中文显示的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# --- 2. 模拟FAST反射面的几何数据 ---
print("正在根据数学模型生成反射面三维数据...")

# 根据题目参数定义基准球面
R = 300.4  # 球面半径 (米)
aperture_diameter = 300 # 口径 (米)

# 我们将使用球面坐标系来生成点，然后转换为笛卡尔坐标系
# phi: 从Z轴正方向开始的极角 (polar angle)
# theta: 在XY平面上的方位角 (azimuthal angle)

# 创建角度网格
theta = np.linspace(0, 2 * np.pi, 100) # 方位角从 0 到 360 度

# 计算形成300米口径所需的极角范围
# 口径 D = 2 * R * sin(phi_max) => phi_max = asin(D / (2R))
phi_max = np.arcsin(aperture_diameter / (2 * R))
phi = np.linspace(0, phi_max, 50) # 极角从 0 到 phi_max

# 将角度网格化
theta, phi = np.meshgrid(theta, phi)

# 球面坐标到笛卡尔坐标的转换
X = R * np.sin(phi) * np.cos(theta)
Y = R * np.sin(phi) * np.sin(theta)
Z = R * np.cos(phi)

# 为了让碗口朝上，我们将Z坐标进行翻转
# 使其顶点位于Z=0，碗口朝向Z轴正方向
Z = R - Z

print("数据生成完毕。")

# --- 3. 创建并美化三维图像 ---
print("正在绘制三维图像...")

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 垂直夸张，让Z方向看起来不那么扁平（>1 更“高”，=1 保持真实比例）
vertical_exaggeration = 3.0
Z_plot = Z * vertical_exaggeration

# 使用线框绘制蓝色栅格（无填充颜色）
wire = ax.plot_wireframe(X, Y, Z_plot, rstride=2, cstride=2, color='blue', linewidth=0.6)

# 线框模式不需要颜色条

# 设置标题和坐标轴标签
ax.set_title('FAST 望远镜反射面三维数学模型', fontsize=20, pad=20)
ax.set_xlabel('X 坐标', fontsize=12, labelpad=10)
ax.set_ylabel('Y 坐标', fontsize=12, labelpad=10)
ax.set_zlabel('Z 坐标', fontsize=12, labelpad=10)
# 保留网格，仅隐藏刻度线与刻度标签
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.zaxis.set_major_formatter(NullFormatter())
ax.tick_params(axis='x', which='both', length=0)
ax.tick_params(axis='y', which='both', length=0)
ax.tick_params(axis='z', which='both', length=0)

# --- 核心修正：设置真实的长宽比，避免视觉失真 ---
# 计算数据在各个轴上的范围（Z 使用夸张后的 Z_plot）
x_range = X.max() - X.min()
y_range = Y.max() - Y.min()
z_range = Z_plot.max() - Z_plot.min()
# 找到最大的范围
max_range = np.array([x_range, y_range, z_range]).max()
# 计算各个轴的中心点
x_mid = (X.max() + X.min()) * 0.5
y_mid = (Y.max() + Y.min()) * 0.5
z_mid = (Z_plot.max() + Z_plot.min()) * 0.5
# 以最大范围为基准，统一设置所有轴的显示界限
ax.set_xlim(x_mid - max_range * 0.5, x_mid + max_range * 0.5)
ax.set_ylim(y_mid - max_range * 0.5, y_mid + max_range * 0.5)
ax.set_zlim(z_mid - max_range * 0.5, z_mid + max_range * 0.5)


# 调整视角 (elevation, azimuth)
ax.view_init(elev=20, azim=30)

# 添加网格背景
ax.grid(True)

print("图像绘制完成。")

# 显示图像
plt.show()
