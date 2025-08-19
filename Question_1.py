import numpy as np

# --- 1. 定义常量和初始参数 ---
# 根据问题描述设置常量
R = 300.4  # 球面半径
z_F = 160.4136  # 抛物面焦点z坐标

# 牛顿法初始值
p0 = 139.9864

num_points = 5000  # 模拟点的数量
max_radius = 150.0  # 模拟区域的最大半径

# 在圆形区域内随机生成点
r = np.sqrt(np.random.uniform(0, max_radius**2, num_points))
theta = np.random.uniform(0, 2 * np.pi, num_points)
xi = r * np.cos(theta)
yi = r * np.sin(theta)


r_sq = xi**2 + yi**2

# 计算这些点在初始球面上的z坐标
z_sphere = np.sqrt(R**2 - r_sq)



def J(p):
    """
    目标函数 J(p)，计算偏差平方和。
    """
    if p <= 0:
        return np.inf  # 焦距必须为正
    # 计算在当前焦距p下，各节点在抛物面上的z坐标
    z_paraboloid = z_F + p - r_sq / (4 * p)
    # 返回偏差的平方和
    return np.sum((z_sphere - z_paraboloid)**2)

def J_prime(p, h=1e-6):
    """
    使用中心差分法计算 J(p) 的一阶导数。
    """
    return (J(p + h) - J(p - h)) / (2 * h)

def J_double_prime(p, h=1e-6):
    """
    使用中心差分法计算 J(p) 的二阶导数。
    """
    return (J(p + h) - 2 * J(p) + J(p - h)) / (h**2)

# --- 4. 牛顿法求解 ---
def newton_method(p_init, tol=1e-8, max_iter=100):
    """
    使用牛顿法寻找 J(p) 的最小值。
    
    Args:
        p_init (float): 焦距 p 的初始猜测值。
        tol (float): 收敛的容忍度。
        max_iter (int): 最大迭代次数。
        
    Returns:
        float: 优化后的最佳焦距 p*。
    """
    p_k = p_init
    print("--- 开始牛顿法迭代 ---")
    print(f"初始值 p_0 = {p_k:.8f}")

    for k in range(max_iter):
        # 计算一阶和二阶导数
        J_p = J_prime(p_k)
        J_dp = J_double_prime(p_k)

        # 避免除以一个非常小的数
        if abs(J_dp) < 1e-9:
            print("错误：二阶导数过小，迭代终止。")
            break

        # 牛顿法迭代公式
        p_k_plus_1 = p_k - J_p / J_dp
        
        # 打印当前迭代信息
        step_size = abs(p_k_plus_1 - p_k)
        print(f"迭代次数 {k+1}: p_{k+1} = {p_k_plus_1:.8f}, |Δp| = {step_size:.2e}")
        
        # 检查收敛条件
        if step_size <= tol:
            print(f"\n收敛条件满足，迭代结束。")
            return p_k_plus_1

        p_k = p_k_plus_1
        
    print("\n已达到最大迭代次数但未收敛。")
    return p_k

# --- 5. 执行求解并分析结果 ---
# 运行牛顿法求解最优 p
p_star = newton_method(p0)

# 使用最优解 p* 计算最终结果
J_star = J(p_star)
z_paraboloid_final = z_F + p_star - r_sq / (4 * p_star)
errors = z_sphere - z_paraboloid_final
max_error = np.max(np.abs(errors))
z_v = z_F + p_star # 抛物面顶点z坐标

print("\n--- 最终计算结果 ---")
print(f"最优焦距 p* = {p_star:.4f} m")
print(f"目标函数最小值 J(p*) = {J_star:.4f} m^2")
print(f"最大节点偏差 max|e_i| = {max_error:.4f} m")

print("\n最终理想抛物面方程:")
denominator = 4 * p_star
print(f"z = {z_v:.4f} - (x^2 + y^2) / {denominator:.4f}")

print("\n顶点 V 坐标:")
print(f"(0, 0, {z_v:.4f}) m")
print("\n焦点 F 坐标:")
print(f"(0, 0, {z_F:.4f}) m")