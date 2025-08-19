import sys
import os
from typing import Tuple, Optional
import argparse
import io

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.optimize import minimize, NonlinearConstraint, Bounds
    import scipy.sparse as sp
except Exception as exc:  # noqa: BLE001
    raise exc


class ParaboloidFitModel:
    """
    抛物面拟合的数学模型封装。

    变量：d ∈ R^N（每个主索节点的促动器伸缩量）

    目标函数：
        J(d) = Σ_i [ (u·r'_i)^2 + (v·r'_i)^2 - 4 p (n·r'_i - z_v) ]^2
        其中 r'_i = r_i + (d_i/R) r_i = (1 + d_i/R) r_i

    约束：
      1) 促动器行程：-0.6 ≤ d_i ≤ 0.6
      2) 下拉索长度不变（等式约束）：
           || r'_i - a_i || = || r_i - a_i ||，等价于
           || r'_i - a_i ||^2 - || r_i - a_i ||^2 = 0
      3) 相邻节点距离变化（不等式约束）：
           || r'_i - r'_j || - 1.0007 || r_i - r_j || ≤ 0 （对每条相邻边 (i,j)）
    """

    def __init__(
        self,
        r: NDArray[np.float64],
        a: NDArray[np.float64],
        edges: NDArray[np.int64],
        n: NDArray[np.float64],
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        P: NDArray[np.float64],
        V: NDArray[np.float64],
        R: Optional[float] = None,
    ) -> None:
        # 基本形状与检查
        assert r.ndim == 2 and r.shape[1] == 3, "r 应为 Nx3 数组"
        assert a.shape == r.shape, "a 形状必须与 r 相同"
        assert edges.ndim == 2 and edges.shape[1] == 2, "edges 应为 Mx2 整数数组"
        assert n.shape == (3,) and u.shape == (3,) and v.shape == (3,), "n/u/v 均应为长度为3的向量"
        assert P.shape == (3,) and V.shape == (3,), "P、V 均应为长度为3的向量"

        self.r: NDArray[np.float64] = r.astype(np.float64)
        self.a: NDArray[np.float64] = a.astype(np.float64)
        self.edges: NDArray[np.int64] = edges.astype(np.int64)
        self.n: NDArray[np.float64] = n / np.linalg.norm(n)
        # 确保 u、v 与 n 正交且归一（若提供的 u、v 已满足，此步骤仅做微调）
        self.u: NDArray[np.float64] = self._orthonormalize(u, self.n)
        self.v: NDArray[np.float64] = self._orthonormalize(v, self.n, orth_to=self.u)

        # 半径 R：若未指定，则用 |r_i| 的均值近似整个结构的名义半径
        if R is None:
            norms = np.linalg.norm(self.r, axis=1)
            self.R: float = float(np.mean(norms))
        else:
            self.R = float(R)

        # 抛物面参数
        self.P: NDArray[np.float64] = P.astype(np.float64)
        self.V: NDArray[np.float64] = V.astype(np.float64)
        self.p: float = float(np.linalg.norm(self.P - self.V))
        self.z_v: float = float(np.dot(self.V, self.n))

        # 预计算静态量：初始边长 || r_i - r_j ||
        i_idx = self.edges[:, 0]
        j_idx = self.edges[:, 1]
        self.initial_edge_lengths: NDArray[np.float64] = np.linalg.norm(
            self.r[i_idx] - self.r[j_idx], axis=1
        )

        # 预计算下拉索初始长度平方 || r_i - a_i ||^2
        self.anchor_base_len_sq: NDArray[np.float64] = np.sum(
            (self.r - self.a) ** 2, axis=1
        )

        # 变量维度
        self.N: int = self.r.shape[0]
        self.M_edges: int = self.edges.shape[0]

    @staticmethod
    def _orthonormalize(
        vec: NDArray[np.float64],
        axis: NDArray[np.float64],
        orth_to: Optional[NDArray[np.float64]] = None,
        eps: float = 1e-12,
    ) -> NDArray[np.float64]:
        """
        将 vec 投影到与 axis 正交的空间，并与 orth_to 再次正交；最后归一化。
        用于得到与 n 正交的 u、v（且 u ⟂ v）。
        """
        v = vec - np.dot(vec, axis) * axis
        if orth_to is not None:
            v = v - np.dot(v, orth_to) * orth_to
        nv = np.linalg.norm(v)
        if nv < eps:
            # 如果退化，则任取一个与 axis 正交的单位向量
            tmp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(tmp, axis)) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0])
            v = tmp - np.dot(tmp, axis) * axis
            v = v / np.linalg.norm(v)
        else:
            v = v / nv
        return v

    # =============== 工具：从 d 计算 r' 与缩放因子 s = 1 + d/R ===============
    def _scaled_positions(self, d: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        s = 1.0 + d / self.R  # 形如 (N,)
        r_prime = self.r * s[:, None]
        return r_prime, s

    # ============================ 目标函数与梯度 ============================
    def objective(self, d: NDArray[np.float64]) -> float:
        """
        标量目标函数 J(d)。
        """
        r_prime, _ = self._scaled_positions(d)

        # 各项内积（矢量化计算）
        alpha = r_prime @ self.u  # (N,)
        beta = r_prime @ self.v   # (N,)
        gamma = r_prime @ self.n  # (N,)

        term = (alpha ** 2) + (beta ** 2) - 4.0 * self.p * (gamma - self.z_v)
        J = float(np.sum(term ** 2))
        return J

    def objective_grad(self, d: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        目标函数的梯度 ∇J(d)。
        利用解析形式：
            r'_i = (1 + d_i/R) r_i，dr'_i/dd_i = (1/R) r_i。
            term_i = (u·r'_i)^2 + (v·r'_i)^2 - 4p (n·r'_i - z_v)
            d term_i / dd_i = (2/R)[ (u·r'_i)(u·r_i) + (v·r'_i)(v·r_i) ] - (4p/R)(n·r_i)
            dJ/dd_i = 2 term_i * d term_i / dd_i
        """
        r_prime, _ = self._scaled_positions(d)

        alpha = r_prime @ self.u
        beta = r_prime @ self.v
        gamma = r_prime @ self.n

        term = (alpha ** 2) + (beta ** 2) - 4.0 * self.p * (gamma - self.z_v)

        # 预计算固定内积 (u·r_i), (v·r_i), (n·r_i)
        u_dot_r = self.r @ self.u
        v_dot_r = self.r @ self.v
        n_dot_r = self.r @ self.n

        dterm = (2.0 / self.R) * (alpha * u_dot_r + beta * v_dot_r) - (4.0 * self.p / self.R) * n_dot_r
        grad = 2.0 * term * dterm
        return grad.astype(np.float64)

    # ============================ 等式约束（下拉索长度不变） ============================
    def anchor_constraint(self, d: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        f_i(d) = || r'_i - a_i ||^2 - || r_i - a_i ||^2 = 0，i=1..N
        向量返回值形如 (N,)
        """
        r_prime, _ = self._scaled_positions(d)
        diff = r_prime - self.a
        val = np.sum(diff * diff, axis=1) - self.anchor_base_len_sq
        return val

    def anchor_constraint_jac(self, d: NDArray[np.float64]):  # -> sp.spmatrix
        """
        雅可比为对角稀疏矩阵：
            ∂f_i/∂d_i = (2/R) ( (r'_i - a_i) · r_i )，其余为 0。
        """
        r_prime, _ = self._scaled_positions(d)
        diag_vals = (2.0 / self.R) * np.einsum('ij,ij->i', (r_prime - self.a), self.r)
        J = sp.diags(diag_vals, format='csr')
        return J

    # ============================ 不等式约束（相邻点距离） ============================
    def edge_constraint(self, d: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        c_k(d) = || r'_i - r'_j || - 1.0007 || r_i - r_j || ≤ 0
        按行对应每条边 (i,j)。
        """
        r_prime, _ = self._scaled_positions(d)
        i_idx = self.edges[:, 0]
        j_idx = self.edges[:, 1]

        delta = r_prime[i_idx] - r_prime[j_idx]
        lens = np.linalg.norm(delta, axis=1)
        val = lens - 1.0007 * self.initial_edge_lengths
        return val

    def edge_constraint_jac(self, d: NDArray[np.float64]):  # -> sp.spmatrix
        """
        雅可比仅在列 i 与 j 处非零：
            ∂c/∂d_i = (Δ'·r_i) / (R ||Δ'||)
            ∂c/∂d_j = -(Δ'·r_j) / (R ||Δ'||)
        其中 Δ' = r'_i - r'_j
        """
        r_prime, _ = self._scaled_positions(d)
        i_idx = self.edges[:, 0]
        j_idx = self.edges[:, 1]

        delta = r_prime[i_idx] - r_prime[j_idx]  # (M,3)
        lens = np.linalg.norm(delta, axis=1)

        # 防止除零
        eps = 1e-12
        safe_lens = np.maximum(lens, eps)

        # 计算每条边对应的导数值（两列）
        dot_i = np.einsum('ij,ij->i', delta, self.r[i_idx])  # Δ' · r_i
        dot_j = np.einsum('ij,ij->i', delta, self.r[j_idx])  # Δ' · r_j

        val_i = (dot_i / (self.R * safe_lens))  # ∂c/∂d_i
        val_j = -(dot_j / (self.R * safe_lens))  # ∂c/∂d_j

        # 构造稀疏矩阵（M x N），每行仅两个非零
        rows = np.repeat(np.arange(self.edges.shape[0]), 2)
        cols = np.concatenate([i_idx, j_idx])
        data = np.concatenate([val_i, val_j])

        J = sp.coo_matrix((data, (rows, cols)), shape=(self.edges.shape[0], self.N)).tocsr()
        return J


def solve_paraboloid_fit(
    r: NDArray[np.float64],
    a: NDArray[np.float64],
    edges: NDArray[np.int64],
    n: NDArray[np.float64],
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    P: NDArray[np.float64],
    V: NDArray[np.float64],
    R: Optional[float] = None,
    d0: Optional[NDArray[np.float64]] = None,
    bounds: Tuple[float, float] = (-0.6, 0.6),
    max_iter: int = 500,
) -> Tuple[NDArray[np.float64], dict]:
    """
    使用 trust-constr 求解器求解上述优化问题。

    返回：
      - d_opt: 最优解（N,）
      - info: 结果信息（dict）
    """
    model = ParaboloidFitModel(r=r, a=a, edges=edges, n=n, u=u, v=v, P=P, V=V, R=R)

    N = r.shape[0]
    if d0 is None:
        d0 = np.zeros(N, dtype=np.float64)

    lb, ub = bounds
    var_bounds = Bounds(lb=np.full(N, lb), ub=np.full(N, ub))

    # 等式约束：下拉索长度不变
    anchor_cons = NonlinearConstraint(
        fun=model.anchor_constraint,
        lb=np.zeros(N, dtype=np.float64),
        ub=np.zeros(N, dtype=np.float64),
        jac=model.anchor_constraint_jac,
    )

    # 不等式约束：相邻距离不超过 1.0007 倍
    edge_cons = NonlinearConstraint(
        fun=model.edge_constraint,
        lb=-np.inf * np.ones(model.M_edges, dtype=np.float64),
        ub=np.zeros(model.M_edges, dtype=np.float64),
        jac=model.edge_constraint_jac,
    )

    res = minimize(
        fun=model.objective,
        x0=d0,
        jac=model.objective_grad,
        method="trust-constr",
        bounds=var_bounds,
        constraints=[anchor_cons, edge_cons],
        options=dict(
            maxiter=max_iter,
            verbose=3,  # 输出迭代日志
            xtol=1e-10,
            gtol=1e-8,
            barrier_tol=1e-8,
        ),
    )

    info = {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "nit": int(res.niter),
        "final_objective": float(res.fun),
        "optimality": float(getattr(res, "optimality", np.nan)),
        "constr_violation": float(getattr(res, "constr_violation", np.nan)),
    }

    return res.x.astype(np.float64), info


# ============================ 数据加载与演示 ============================
def try_load_csv_triplet(
    base_dir: str,
    r_file: str = "r.csv",
    a_file: str = "a.csv",
    edges_file: str = "edges.csv",
) -> Optional[Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]]:
    """
    从 CSV 文件加载 r、a、edges。
    - r.csv / a.csv: 每行三个数（x,y,z）
    - edges.csv: 每行两个整数（i,j），索引默认 0-based；若检测到最大索引为 N，则自动转为 0-based。
    """
    r_path = os.path.join(base_dir, r_file)
    a_path = os.path.join(base_dir, a_file)
    e_path = os.path.join(base_dir, edges_file)

    if not (os.path.isfile(r_path) and os.path.isfile(a_path) and os.path.isfile(e_path)):
        return None

    def load_float_matrix(path: str) -> NDArray[np.float64]:
        arr = np.loadtxt(path, delimiter=",")
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr.astype(np.float64)

    r = load_float_matrix(r_path)
    a = load_float_matrix(a_path)
    edges = np.loadtxt(e_path, delimiter=",", dtype=np.int64)
    if edges.ndim == 1:
        edges = edges[None, :]

    # 自动检测 1-based -> 0-based
    N = r.shape[0]
    if np.max(edges) >= N:
        edges = edges - 1

    return r, a, edges


def generate_demo_data(N: int = 40, radius: float = 10.0, seed: int = 42):
    """
    生成一组小规模演示数据，使得 d=0 时严格可行：
      - 节点：随机分布在半径为 radius 的球面上
      - 下拉索锚点：a_i = r_i - c·(r_i/||r_i||)，保证初始长度固定
      - 邻接：构造一个环（每点连前后两个点），保证初始边长已知
      - 轴向：n 为 z 轴，u/v 与 n 正交
      - 抛物面参数：V=0，P 在 z 轴上，使 p>0
    """
    rng = np.random.default_rng(seed)

    # 在球面上均匀采样方向
    phi = rng.uniform(0.0, 2.0 * np.pi, size=N)
    costheta = rng.uniform(-1.0, 1.0, size=N)
    sintheta = np.sqrt(1.0 - costheta**2)
    directions = np.stack(
        [np.cos(phi) * sintheta, np.sin(phi) * sintheta, costheta], axis=1
    )

    r = radius * directions  # Nx3

    # 构造环形边
    edges = np.stack([np.arange(N), (np.arange(N) + 1) % N], axis=1).astype(np.int64)

    # 下拉索锚点：沿径向向内偏移固定长度 c
    c = 1.0
    a = r - c * (r / np.linalg.norm(r, axis=1, keepdims=True))

    # 抛物面参数
    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    V = np.zeros(3, dtype=np.float64)
    P = np.array([0.0, 0.0, radius], dtype=np.float64)  # 令 p = radius

    return r, a, edges, n, u, v, P, V, radius


def axis_from_alpha_beta(alpha_deg: float, beta_deg: float) -> NDArray[np.float64]:
    """
    根据天体方位角 α（deg）、高度角 β（deg）计算入射方向的反向（抛物面轴方向）单位向量 n。
    约定：α 从 x 轴起，逆时针至 y 轴（右手系，地平坐标）；β 为高度角。
    对于 α=0°, β=90°，应得到 n≈(0,0,1)。
    """
    rad = np.pi / 180.0
    alpha = alpha_deg * rad
    beta = beta_deg * rad
    # 入射方向（来自天体）为 s = (cosβ cosα, cosβ sinα, sinβ)
    # 抛物面轴方向取与入射方向相反的单位向量，或直接取 n = +z 当 β=90°。
    s = np.array([np.cos(beta) * np.cos(alpha), np.cos(beta) * np.sin(alpha), np.sin(beta)], dtype=np.float64)
    n = s / np.linalg.norm(s)
    return n


def construct_uv_from_n(n: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    给定轴向 n，构造与之正交的单位向量 u、v。
    """
    # 选择一个不平行于 n 的基向量
    trial = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(trial, n)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = trial - np.dot(trial, n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return u, v


def find_ideal_paraboloid_zenith(
    r: NDArray[np.float64],
    a: NDArray[np.float64],
    edges: NDArray[np.int64],
    alpha_deg: float = 0.0,
    beta_deg: float = 90.0,
    p_bounds: Tuple[float, float] = (50.0, 500.0),
    zv_bounds: Tuple[float, float] = (-500.0, 100.0),
    inner_max_iter: int = 120,
    grid_evals: int = 5,
) -> Tuple[Tuple[float, float], Tuple[NDArray[np.float64], dict]]:
    """
    在 α=0°, β=90°（或其它给定 α,β）条件下，联合考虑面板调节，搜索最优抛物面参数 (p, z_v)。
    - 外层：对 (p, z_v) 做粗网格搜索以降低计算量
    - 内层：每个 (p, z_v) 固定时，调用 trust-constr 优化 d
    返回：(p*, z_v*), (d_opt, info)
    """
    n = axis_from_alpha_beta(alpha_deg, beta_deg)
    u, v = construct_uv_from_n(n)

    # 粗网格搜索
    p_candidates = np.linspace(p_bounds[0], p_bounds[1], num=max(2, grid_evals))
    zv_candidates = np.linspace(zv_bounds[0], zv_bounds[1], num=max(2, grid_evals))

    best_val = np.inf
    best_p, best_zv = None, None
    best_solution = None

    R_guess = float(np.linalg.norm(r, axis=1).mean())

    for p in p_candidates:
        for zv in zv_candidates:
            V = zv * n
            P = V + p * n
            try:
                d_opt, info = solve_paraboloid_fit(
                    r=r,
                    a=a,
                    edges=edges,
                    n=n,
                    u=u,
                    v=v,
                    P=P,
                    V=V,
                    R=R_guess,
                    d0=np.zeros(r.shape[0], dtype=np.float64),
                    bounds=(-0.6, 0.6),
                    max_iter=inner_max_iter,
                )
                val = info.get("final_objective", np.inf)
            except Exception:
                val = np.inf
                d_opt, info = None, {"success": False}

            if val < best_val and info.get("success", False):
                best_val = val
                best_p, best_zv = float(p), float(zv)
                best_solution = (d_opt, info)

    if best_solution is None:
        raise RuntimeError("未能在给定搜索范围内找到可行的抛物面参数 (p, z_v)。请扩大搜索区间或放宽内层迭代次数。")

    return (best_p, best_zv), best_solution


def main() -> None:
    # 优先尝试从附件文件加载（附件2：锚点与标准态节点；附件3：三角单元 -> 边集）
    base_dir = os.path.dirname(os.path.abspath(__file__))

    def _read_csv_text(path: str) -> Tuple[str, str]:
        # 先以二进制读取，再尝试多种常见编码解码
        with open(path, "rb") as fb:
            raw = fb.read()
        for enc in ("utf-8-sig", "gbk", "gb18030", "cp936", "utf-8"):
            try:
                return raw.decode(enc), enc
            except Exception:
                continue
        # 最后一次尝试严格 utf-8 以触发明确异常
        raw.decode("utf-8")
        return "", "utf-8"

    def load_from_attachments(base_dir_: str):
        nodes_path = os.path.join(base_dir_, "附件2.csv")
        tri_path = os.path.join(base_dir_, "附件3.csv")
        if not (os.path.isfile(nodes_path) and os.path.isfile(tri_path)):
            return None

        # 读取节点（锚点 a 与标准态上节点 r）
        text, enc = _read_csv_text(nodes_path)
        import csv
        reader = csv.reader(io.StringIO(text))
        header = next(reader, None)
        name_to_idx = {}
        anchors = []
        tops = []
        names = []
        for row in reader:
            if not row or len(row) < 7:
                continue
            name = row[0].strip()
            try:
                ax, ay, az = float(row[1]), float(row[2]), float(row[3])
                tx, ty, tz = float(row[4]), float(row[5]), float(row[6])
            except ValueError:
                # 略过非数据行
                continue
            idx = len(names)
            name_to_idx[name] = idx
            names.append(name)
            anchors.append([ax, ay, az])
            tops.append([tx, ty, tz])
        # no file handle to close (StringIO)

        if len(names) == 0:
            return None

        a_arr = np.asarray(anchors, dtype=np.float64)
        r_arr = np.asarray(tops, dtype=np.float64)

        # 读取三角面并生成无向边集
        text2, enc2 = _read_csv_text(tri_path)
        reader2 = csv.reader(io.StringIO(text2))
        header2 = next(reader2, None)
        edge_set = set()
        missing_names = set()
        for row in reader2:
            if not row or len(row) < 3:
                continue
            n1, n2, n3 = row[0].strip(), row[1].strip(), row[2].strip()
            if n1 not in name_to_idx or n2 not in name_to_idx or n3 not in name_to_idx:
                missing_names.update(x for x in (n1, n2, n3) if x not in name_to_idx)
                continue
            i1, i2, i3 = name_to_idx[n1], name_to_idx[n2], name_to_idx[n3]
            for (u_, v_) in ((i1, i2), (i2, i3), (i1, i3)):
                if u_ == v_:
                    continue
                e = (u_, v_) if u_ < v_ else (v_, u_)
                edge_set.add(e)
        # no file handle to close (StringIO)

        if missing_names:
            print(f"警告：三角单元中存在 {len(missing_names)} 个未出现在节点表的名称，将被忽略。示例：{list(sorted(missing_names))[:5]}")

        edges_arr = np.asarray(sorted(edge_set), dtype=np.int64)
        return r_arr, a_arr, edges_arr, names

    loaded = load_from_attachments(base_dir)
    if loaded is not None:
        r, a, edges, node_names = loaded
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        # 依据数据自定一个抛物面参数（可按需要修改）：
        gamma = r @ n
        z_v_val = float(np.mean(gamma))
        p_val = float(abs(z_v_val))  # 取与量级相当的焦距
        V = z_v_val * n
        P = V + p_val * n
        R = float(np.linalg.norm(r, axis=1).mean())
        print("已从附件加载数据：", r.shape[0], "个节点，", edges.shape[0], "条边。编码自动识别。")
    else:
        # 回退：若附件不存在，则尝试 r/a/edges 常规三件套；仍不存在则用演示数据
        fallback = try_load_csv_triplet(base_dir)
        if fallback is not None:
            r, a, edges = fallback
            n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            V = np.zeros(3, dtype=np.float64)
            P = np.array([0.0, 0.0, np.linalg.norm(r, axis=1).mean()], dtype=np.float64)
            R = float(np.linalg.norm(r, axis=1).mean())
            print("已从 CSV 加载数据：", r.shape[0], "个节点，", edges.shape[0], "条边")
        else:
            r, a, edges, n, u, v, P, V, R = generate_demo_data(N=60, radius=8.0)
            print("未找到 CSV/附件数据，使用演示数据：", r.shape[0], "个节点，", edges.shape[0], "条边")

    parser = argparse.ArgumentParser(description="抛物面拟合求解器")
    parser.add_argument("--zenith-opt", action="store_true", help="在 α=0°, β=90° 条件下联立搜索最优 (p, z_v)")
    parser.add_argument("--alpha", type=float, default=0.0, help="方位角 α (deg)")
    parser.add_argument("--beta", type=float, default=90.0, help="高度角 β (deg)")
    parser.add_argument("--p-min", type=float, default=50.0)
    parser.add_argument("--p-max", type=float, default=500.0)
    parser.add_argument("--zv-min", type=float, default=-500.0)
    parser.add_argument("--zv-max", type=float, default=100.0)
    parser.add_argument("--outer-grid", type=int, default=5, help="(p,z_v) 粗网格划分数")
    parser.add_argument("--inner-iters", type=int, default=150, help="内层 d 优化最大迭代数")
    args = parser.parse_args(args=sys.argv[1:])

    if args.zenith_opt:
        # α,β 转换为轴向 n，随后进行外层网格搜索
        (p_star, zv_star), (d_opt, info) = find_ideal_paraboloid_zenith(
            r=r,
            a=a,
            edges=edges,
            alpha_deg=args.alpha,
            beta_deg=args.beta,
            p_bounds=(args.p_min, args.p_max),
            zv_bounds=(args.zv_min, args.zv_max),
            inner_max_iter=int(args.inner_iters),
            grid_evals=int(args.outer_grid),
        )

        n = axis_from_alpha_beta(args.alpha, args.beta)
        V = zv_star * n
        P = V + p_star * n
        print("最优抛物面参数：")
        print(f"  轴向 n = {n}")
        print(f"  顶点 V = {V}")
        print(f"  焦点 P = {P}")
        print(f"  p = |P-V| = {p_star:.6f},  z_v = V·n = {zv_star:.6f}")
        print("内层求解信息：")
        for k, v in info.items():
            print(f"  {k}: {v}")

        out_path = os.path.join(base_dir, "d_opt.csv")
        np.savetxt(out_path, d_opt, delimiter=",", fmt="%.10f")
        print(f"最优解已保存至: {out_path}")

        try:
            node_names  # type: ignore[name-defined]
            name_out = os.path.join(base_dir, "d_opt_with_names.csv")
            with open(name_out, "w", encoding="utf-8") as f:
                f.write("name,d\n")
                for name, di in zip(node_names, d_opt):
                    f.write(f"{name},{di:.10f}\n")
            print(f"已额外保存带节点名的结果: {name_out}")
        except NameError:
            pass
    else:
        # 仅固定 (n,u,v,P,V) 情况下求解 d
        d_opt, info = solve_paraboloid_fit(
            r=r,
            a=a,
            edges=edges,
            n=n,
            u=u,
            v=v,
            P=P,
            V=V,
            R=R,
            d0=np.zeros(r.shape[0], dtype=np.float64),
            bounds=(-0.6, 0.6),
            max_iter=200,
        )

        print("求解结果：")
        for k, v in info.items():
            print(f"  {k}: {v}")

        out_path = os.path.join(base_dir, "d_opt.csv")
        np.savetxt(out_path, d_opt, delimiter=",", fmt="%.10f")
        print(f"最优解已保存至: {out_path}")

    # 若包含节点名，则额外保存带名结果，便于检索
    try:
        node_names  # type: ignore[name-defined]
        name_out = os.path.join(base_dir, "d_opt_with_names.csv")
        with open(name_out, "w", encoding="utf-8") as f:
            f.write("name,d\n")
            for name, di in zip(node_names, d_opt):
                f.write(f"{name},{di:.10f}\n")
        print(f"已额外保存带节点名的结果: {name_out}")
    except NameError:
        pass


if __name__ == "__main__":
    main()

