import numpy as np
import pandas as pd
from scipy.optimize import minimize
from itertools import combinations

print("--- 模型二求解代码 (基于附录数据结构) ---")

# --- 1. 定义常量和天体方位 ---
R = 300.4  # 球面半径
F = 0.466 * R  # 焦距 (p)
alpha_deg = 36.795
beta_deg = 78.169


try:
    print("正在加载附件数据...")
    # 读取CSV文件并指定gbk编码
    nodes_df = pd.read_csv("附件1.csv", encoding='gbk')
    actuators_df = pd.read_csv("附件2.csv", encoding='gbk')
    panels_df = pd.read_csv("附件3.csv", encoding='gbk')

    # 读取后，将第一列设置为索引，以避免列名不匹配问题
    nodes_df.set_index(nodes_df.columns[0], inplace=True)
    actuators_df.set_index(actuators_df.columns[0], inplace=True)
    
    print("附件数据加载完毕。")
    # 打印列名以供调试检查
    print(f"附件1检测到的列名: {nodes_df.columns.tolist()}")
    print(f"附件2检测到的列名: {actuators_df.columns.tolist()}")

except FileNotFoundError as e:
    print(f"错误：找不到文件 {e.filename}。请确保附件文件存在且文件名正确。")
    exit()
except Exception as e:
    print(f"读取文件时发生错误: {e}")
    exit()


# --- 从数据中提取邻接关系 ---
print("正在根据面板数据构建节点邻接关系...")
neighbor_pairs = set()
panel_node_columns = ['主索节点1', '主索节点2', '主索节点3']
for index, row in panels_df.iterrows():
    panel_node_ids = pd.to_numeric(row[panel_node_columns], errors='coerce').dropna().astype(int).tolist()
    for pair in combinations(panel_node_ids, 2):
        neighbor_pairs.add(tuple(sorted(pair)))
print(f"成功构建了 {len(neighbor_pairs)} 对邻接关系。")


num_opt_nodes = 100 
np.random.seed(0)
all_node_ids = nodes_df.index.tolist()
sample_node_ids = np.random.choice(all_node_ids, num_opt_nodes, replace=False)


nodes_to_optimize = nodes_df.loc[sample_node_ids, ['X坐标(米)', 'Y坐标(米)', 'Z坐标(米)']].values
anchors_to_optimize = actuators_df.loc[sample_node_ids, ['下端点x坐标', '下端点y坐标', '下端点z坐标']].values

original_id_to_sample_idx = {node_id: i for i, node_id in enumerate(sample_node_ids)}
neighbor_pairs_optimized = [
    (original_id_to_sample_idx[p[0]], original_id_to_sample_idx[p[1]])
    for p in neighbor_pairs
    if p[0] in original_id_to_sample_idx and p[1] in original_id_to_sample_idx
]
print(f"将对 {num_opt_nodes} 个抽样节点和 {len(neighbor_pairs_optimized)} 对邻接关系进行优化。")

# --- 3. 建立理想抛物面几何模型 ---
alpha = np.deg2rad(alpha_deg)
beta = np.deg2rad(beta_deg)
n = np.array([np.cos(beta) * np.sin(alpha), np.cos(beta) * np.cos(alpha), np.sin(beta)])
V = R * n # 理想抛物面顶点坐标
e_x = np.array([1.0, 0.0, 0.0])
u0 = e_x - np.dot(n, e_x) * n
u = u0 / np.linalg.norm(u0)
v = np.cross(n, u)

# --- 4. 定义目标函数和约束函数 ---
def objective_function(d, nodes):
    r_prime = nodes - d[:, np.newaxis] * (nodes / R)
    r_dot_u = np.dot(r_prime, u)
    r_dot_v = np.dot(r_prime, v)
    r_dot_n = np.dot(r_prime, n)
    residuals = r_dot_u**2 + r_dot_v**2 - 4 * F * (r_dot_n - R)
    return np.sum(residuals**2)

def cable_length_constraint(d, initial_nodes, anchors):
    initial_dist_sq = np.sum((initial_nodes - anchors)**2, axis=1)
    r_prime = initial_nodes - d[:, np.newaxis] * (initial_nodes / R)
    final_dist_sq = np.sum((r_prime - anchors)**2, axis=1)
    return final_dist_sq - initial_dist_sq

def neighbor_distance_constraint(d, initial_nodes, pairs):
    constraints = []
    r_prime = initial_nodes - d[:, np.newaxis] * (initial_nodes / R)
    for i, j in pairs:
        initial_dist = np.linalg.norm(initial_nodes[i] - initial_nodes[j])
        final_dist = np.linalg.norm(r_prime[i] - r_prime[j])
        constraints.append(0.0007 * initial_dist - (final_dist - initial_dist))
        constraints.append(0.0007 * initial_dist + (final_dist - initial_dist))
    return np.array(constraints)

# --- 5. 设置并运行优化器 ---
d_initial_guess = np.zeros(num_opt_nodes)
actuator_bounds = [(-0.6, 0.6) for _ in range(num_opt_nodes)]
constraints_list = [
    {'type': 'eq', 'fun': cable_length_constraint, 'args': (nodes_to_optimize, anchors_to_optimize)},
    {'type': 'ineq', 'fun': neighbor_distance_constraint, 'args': (nodes_to_optimize, neighbor_pairs_optimized)}
]

print("\n--- 开始执行约束优化 (这可能需要几分钟) ---")
result = minimize(
    fun=objective_function, x0=d_initial_guess, args=(nodes_to_optimize,),
    method='SLSQP', bounds=actuator_bounds, constraints=constraints_list,
    options={'disp': True, 'maxiter': 200, 'ftol': 1e-7}
)

# --- 6. 显示优化结果并保存到Excel文件 ---
print("\n--- 优化结果分析 ---")
if result.success:
    optimal_d = result.x
    final_J = result.fun
    rmse = np.sqrt(final_J / num_opt_nodes)
    
    print(f"优化成功：{result.message}")
    print(f"最终目标函数值 J_min = {final_J:.4e}")
    print(f"节点的均方根误差(RMSE) ≈ {rmse * 1000:.2f} mm")
    print(f"促动器伸缩量范围: [{np.min(optimal_d):.4f} m, {np.max(optimal_d):.4f} m]")
    
    print("\n--- 正在将结果保存到 'result.xlsx' ---")
    try:
        final_nodes = nodes_to_optimize - optimal_d[:, np.newaxis] * (nodes_to_optimize / R)
        results_df = pd.DataFrame({
            '初始X坐标(米)': nodes_to_optimize[:, 0], '初始Y坐标(米)': nodes_to_optimize[:, 1], '初始Z坐标(米)': nodes_to_optimize[:, 2],
            '促动器伸缩量d(m)': optimal_d,
            '调节后X坐标(米)': final_nodes[:, 0], '调节后Y坐标(米)': final_nodes[:, 1], '调节后Z坐标(米)': final_nodes[:, 2]
        }, index=sample_node_ids)
        results_df.index.name = '主索节点编号'
        
        vertex_df = pd.DataFrame({'顶点X坐标(米)': [V[0]], '顶点Y坐标(米)': [V[1]], '顶点Z坐标(米)': [V[2]]})

        with pd.ExcelWriter('result.xlsx') as writer:
            results_df.to_excel(writer, sheet_name='节点调节结果')
            vertex_df.to_excel(writer, sheet_name='抛物面顶点坐标', index=False)
        print("文件 'result.xlsx' 保存成功。")
    except Exception as e:
        print(f"错误：保存Excel文件失败。原因: {e}")
else:
    print(f"优化失败: {result.message}")
