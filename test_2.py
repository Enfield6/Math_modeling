import pandas as pd
import numpy as np
from scipy.optimize import minimize
import openpyxl
from tqdm import tqdm
import warnings

# 忽略优化过程中的一些警告
warnings.filterwarnings("ignore", message="Values in x were outside bounds")

# --- 全局参数定义 ---
R = 300.0  # 基准球面半径 (m)
F_RATIO = 0.466  # 焦径比
F = R * F_RATIO  # 焦距 (m)
R_f = R - F  # 焦面半径 (m)
f_paraboloid = R / 2.0  # 理想抛物面焦距 (m)

# --- 数据加载与预处理 ---
def load_and_prepare_data():
    """
    加载所有附件CSV文件，进行预处理和数据整合。
    """
    print("正在加载和预处理数据...")
    # 加载数据
    nodes_df = pd.read_csv('附件1.csv', header=0)
    actuators_df = pd.read_csv('附件2.csv', header=0)
    panels_df = pd.read_csv('附件3.csv', header=0)

    # 重命名促动器数据中的节点列以便合并
    actuators_df.rename(columns={'对应主索节点编号': '节点编号'}, inplace=True)
    
    # 数据整合
    data = pd.merge(nodes_df, actuators_df, on='节点编号')
    data.set_index('节点编号', inplace=True)

    # 将坐标转换为NumPy数组以便高效计算
    data['P0'] = list([data['P0'].values])
    data = list(data['P0'].values)
    
    # 预计算运动学模型中的不变量
    data['r_hat'] = data.apply(lambda t: t / np.linalg.norm(t))
    data['L_pull'] = data.apply(lambda row: np.linalg.norm(row['P0'] - row), axis=1)

    # 构建主索连接关系和基准长度
    edges = set()
    for _, row in panels_df.iterrows():
        nodes = sorted([row['主索节点1'], row['主索节点2'], row['主索节点3']])
        edges.add(tuple(sorted((nodes, nodes[1]))))
        edges.add(tuple(sorted((nodes[1], nodes[1]))))
        edges.add(tuple(sorted((nodes, nodes[1]))))
    
    edge_list = list(edges)
    edge_lengths = {}
    for n1, n2 in edge_list:
        if n1 in data.index and n2 in data.index:
            p1 = data.loc[n1, 'P0']
            p2 = data.loc[n2, 'P0']
            edge_lengths[(n1, n2)] = np.linalg.norm(p1 - p2)

    print("数据加载与预处理完成。")
    return data, edge_list, edge_lengths, panels_df

# --- 几何模型函数 ---
def get_paraboloid_params(alpha_deg, phi_deg):
    """
    根据天体方位角和仰角计算理想抛物面的几何参数。
    """
    alpha_rad = np.deg2rad(alpha_deg)
    phi_rad = np.deg2rad(phi_deg)

    a_hat = np.array([
        np.cos(phi_rad) * np.cos(alpha_rad),
        np.cos(phi_rad) * np.sin(alpha_rad),
        np.sin(phi_rad)
    ])
    
    P_focus = R_f * a_hat
    V_vertex = P_focus - f_paraboloid * a_hat
    
    return V_vertex, P_focus, a_hat

# --- 问题一求解 ---
def solve_problem_1():
    """
    求解问题一：确定天顶观测时的理想抛物面顶点坐标。
    """
    print("\n--- 正在求解问题一 ---")
    V_vertex_zenith, _, _ = get_paraboloid_params(0, 90)
    print(f"理想抛物面顶点坐标 (X, Y, Z): ({V_vertex_zenith:.4f}, {V_vertex_zenith[1]:.4f}, {V_vertex_zenith[1]:.4f})")
    return V_vertex_zenith

# --- 问题二：优化模型 ---
def solve_problem_2(data, edge_list, edge_lengths):
    """
    求解问题二：离轴观测的反射面优化调节。
    """
    print("\n--- 正在求解问题二 ---")
    
    # 1. 确定理想抛物面
    alpha_deg, phi_deg = 36.795, 78.169
    V_vertex, P_focus, a_hat = get_paraboloid_params(alpha_deg, phi_deg)
    print(f"离轴观测理想抛物面顶点坐标 (X, Y, Z): ({V_vertex:.4f}, {V_vertex[1]:.4f}, {V_vertex[1]:.4f})")

    # 2. 确定优化范围（300m口径内的节点）
    center_proj_on_sphere = R * (V_vertex / np.linalg.norm(V_vertex))
    
    # 计算所有节点到孔径中心的球面距离（近似为欧氏距离，因为区域较小）
    data['dist_to_center'] = data['P0'].apply(lambda p: np.linalg.norm(p - center_proj_on_sphere))
    active_nodes_ids = data[data['dist_to_center'] <= 150.0].index.tolist()
    
    active_data = data.loc[active_nodes_ids]
    active_edges = [(n1, n2) for n1, n2 in edge_list if n1 in active_nodes_ids and n2 in active_nodes_ids]
    
    print(f"已确定优化范围：{len(active_nodes_ids)}个活动节点，{len(active_edges)}根活动主索。")

    # 3. 构建优化问题
    node_map = {node_id: i for i, node_id in enumerate(active_nodes_ids)}
    num_nodes = len(active_nodes_ids)

    # 决策变量: [x1,y1,z1, x2,y2,z2,..., d1,d2,...]
    # 初始猜测值：基准态
    p0_initial = np.concatenate(active_data['P0'].values)
    d0_initial = np.zeros(num_nodes)
    x0 = np.concatenate([p0_initial, d0_initial])

    # 目标函数：最小化节点到理想抛物面的距离平方和
    def objective_function(x):
        P_prime = x[:3*num_nodes].reshape((num_nodes, 3))
        # 抛物面方程: ||X-V||^2 - ((X-V).a)^2 - 4f((X-V).a) = 0
        # 我们最小化这个方程值的平方和
        vec_pv = P_prime - V_vertex
        dot_pv_a = np.einsum('ij,j->i', vec_pv, a_hat)
        errors = np.sum(vec_pv**2, axis=1) - dot_pv_a**2 - 4 * f_paraboloid * dot_pv_a
        return np.sum(errors**2)

    # 约束条件
    # 等式约束：下拉索长度不变
    T0_vec = np.array(active_data.tolist())
    r_hat_vec = np.array(active_data['r_hat'].tolist())
    L_pull_sq_vec = active_data['L_pull'].values**2
    
    def eq_constraints_func(x):
        P_prime = x[:3*num_nodes].reshape((num_nodes, 3))
        d = x[3*num_nodes:]
        T_prime = T0_vec - d[:, np.newaxis] * r_hat_vec
        return np.sum((P_prime - T_prime)**2, axis=1) - L_pull_sq_vec

    # 不等式约束：主索长度变化
    def ineq_constraints_func(x):
        P_prime = x[:3*num_nodes].reshape((num_nodes, 3))
        constraints = []
        for n1_id, n2_id in active_edges:
            i1, i2 = node_map[n1_id], node_map[n2_id]
            p1, p2 = P_prime[i1], P_prime[i2]
            dist_sq = np.sum((p1 - p2)**2)
            l0_sq = edge_lengths[(n1_id, n2_id)]**2
            # g(x) >= 0
            constraints.append(dist_sq - (0.9993**2) * l0_sq)
            constraints.append((1.0007**2) * l0_sq - dist_sq)
        return np.array(constraints)

    # 变量边界
    bounds = [[None, None]] * (3 * num_nodes) + [[-0.6, 0.6]] * num_nodes

    # 定义约束对象
    constraints = [
        {'type': 'eq', 'fun': eq_constraints_func},
        {'type': 'ineq', 'fun': ineq_constraints_func}
    ]
    
    # 4. 运行优化器
    print("开始运行SLSQP优化器，这可能需要几分钟...")
    result = minimize(
        objective_function,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'disp': True, 'maxiter': 100, 'ftol': 1e-6} # 减少迭代次数以获得初步结果
    )

    # 5. 处理并保存结果
    if result.success:
        print("优化成功！")
        final_x = result.x
        final_P_prime = final_x[:3*num_nodes].reshape((num_nodes, 3))
        final_d = final_x[3*num_nodes:]
        
        # 创建结果DataFrames
        res_nodes_df = pd.DataFrame(final_P_prime, columns=['X坐标（米）', 'Y坐标（米）', 'Z坐标（米）'])
        res_nodes_df.insert(0, '节点编号', active_nodes_ids)
        
        res_actuators_df = pd.DataFrame({
            '对应主索节点编号': active_nodes_ids,
            '伸缩量（米）': final_d
        })
        
        res_vertex_df = pd.DataFrame([{'X坐标（米）': V_vertex, 'Y坐标（米）': V_vertex[1], 'Z坐标（米）': V_vertex[1]}])

        # 保存到Excel
        with pd.ExcelWriter('result.xlsx') as writer:
            res_vertex_df.to_excel(writer, sheet_name='理想抛物面顶点坐标', index=False)
            res_nodes_df.to_excel(writer, sheet_name='调整后主索节点编号及坐标', index=False)
            res_actuators_df.to_excel(writer, sheet_name='促动器顶端伸缩量', index=False)
        print("结果已成功保存到 result.xlsx。")
        return res_nodes_df, V_vertex, P_focus, a_hat
    else:
        print("优化失败:", result.message)
        return None, None, None, None

# --- 问题三：光线追踪仿真 ---
def ray_tracing_simulation(node_coords_df, panels_df, P_focus, a_hat, num_rays=100000):
    """
    执行光线追踪仿真以计算接收比。
    """
    print(f"\n开始光线追踪仿真，模拟 {num_rays} 条光线...")
    
    # 准备几何数据
    node_coords = node_coords_df.set_index('节点编号').to_dict('index')
    node_coords = {k: np.array(list(v.values())) for k, v in node_coords.items()}
    
    panel_vertices = []
    for _, row in panels_df.iterrows():
        n1, n2, n3 = row['主索节点1'], row['主索节点2'], row['主索节点3']
        if n1 in node_coords and n2 in node_coords and n3 in node_coords:
            panel_vertices.append([node_coords[n1], node_coords[n2], node_coords[n3]])
    
    panels = np.array(panel_vertices)
    
    # 生成光线
    radius = 150
    rand_r = radius * np.sqrt(np.random.rand(num_rays))
    rand_theta = 2 * np.pi * np.random.rand(num_rays)
    
    # 创建一个正交基
    if abs(np.dot(a_hat, np.array())) < 0.9:
        u_vec = np.cross(a_hat, np.array())
    else:
        u_vec = np.cross(a_hat, np.array())
    u_vec /= np.linalg.norm(u_vec)
    v_vec = np.cross(a_hat, u_vec)
    
    # 光线起始点
    ray_origins_on_disk = (rand_r[:, np.newaxis] * np.cos(rand_theta)[:, np.newaxis] * u_vec +
                           rand_r[:, np.newaxis] * np.sin(rand_theta)[:, np.newaxis] * v_vec)
    ray_origins = ray_origins_on_disk + P_focus - 200 * a_hat # 平移到反射面上方
    ray_directions = np.tile(a_hat, (num_rays, 1))
    
    hits = 0
    for i in tqdm(range(num_rays), desc="光线追踪进度"):
        origin = ray_origins[i]
        direction = ray_directions[i]
        
        min_t = np.inf
        hit_panel_idx = -1

        # Möller–Trumbore 算法: 找到最近的面板交点
        for j, panel in enumerate(panels):
            v0, v1, v2 = panel, panel[1], panel[1]
            edge1, edge2 = v1 - v0, v2 - v0
            h = np.cross(direction, edge2)
            a = np.dot(edge1, h)

            if abs(a) < 1e-7: continue # Ray is parallel to the triangle

            f = 1.0 / a
            s = origin - v0
            u = f * np.dot(s, h)

            if u < 0.0 or u > 1.0: continue

            q = np.cross(s, edge1)
            v = f * np.dot(direction, q)

            if v < 0.0 or u + v > 1.0: continue
            
            t = f * np.dot(edge2, q)
            if t > 1e-7 and t < min_t:
                min_t = t
                hit_panel_idx = j
        
        if hit_panel_idx!= -1:
            # 计算反射
            intersect_point = origin + min_t * direction
            panel = panels[hit_panel_idx]
            normal = np.cross(panel[1] - panel, panel[1] - panel)
            normal /= np.linalg.norm(normal)
            if np.dot(normal, direction) > 0: normal = -normal
            
            refl_dir = direction - 2 * np.dot(direction, normal) * normal
            
            # 与接收平面求交
            denom = np.dot(refl_dir, a_hat)
            if abs(denom) > 1e-7:
                t_refl = np.dot(P_focus - intersect_point, a_hat) / denom
                if t_refl > 0:
                    receiver_intersect = intersect_point + t_refl * refl_dir
                    if np.linalg.norm(receiver_intersect - P_focus) <= 0.5:
                        hits += 1
                        
    return hits / num_rays

def solve_problem_3(data, adjusted_nodes_df, panels_df, V_vertex, P_focus, a_hat):
    """
    求解问题三：计算并比较接收比。
    """
    print("\n--- 正在求解问题三 ---")
    
    # 1. 计算优化后抛物面的接收比
    ratio_paraboloid = ray_tracing_simulation(adjusted_nodes_df, panels_df, P_focus, a_hat)
    print(f"优化后抛物面的接收比: {ratio_paraboloid:.4%}")

    # 2. 计算基准球面的接收比
    active_nodes_ids = adjusted_nodes_df['节点编号'].tolist()
    baseline_nodes_data = data.loc[active_nodes_ids]['P0'].apply(pd.Series)
    baseline_nodes_data.columns = ['X坐标（米）', 'Y坐标（米）', 'Z坐标（米）']
    baseline_nodes_df = baseline_nodes_data.reset_index()

    ratio_sphere = ray_tracing_simulation(baseline_nodes_df, panels_df, P_focus, a_hat)
    print(f"基准球面的接收比: {ratio_sphere:.4%}")

    print("\n--- 性能对比 ---")
    print(f"基准球面接收比: {ratio_sphere:.4%}")
    print(f"优化后抛物面接收比: {ratio_paraboloid:.4%}")
    if ratio_sphere > 0:
        print(f"性能提升倍数: {ratio_paraboloid / ratio_sphere:.2f} 倍")

# --- 主程序 ---
if __name__ == '__main__':
    # 加载数据
    data, edge_list, edge_lengths, panels_df = load_and_prepare_data()
    
    # 求解问题一
    solve_problem_1()
    
    # 求解问题二
    adjusted_nodes_df, V_vertex, P_focus, a_hat = solve_problem_2(data, edge_list, edge_lengths)
    
    # 求解问题三
    if adjusted_nodes_df is not None:
        solve_problem_3(data, adjusted_nodes_df, panels_df, V_vertex, P_focus, a_hat)
    else:
        print("\n问题二优化失败，无法进行问题三的求解。")