import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import logging
from typing import List, Literal, Dict, Tuple
from pydantic import BaseModel, Field, model_validator

# --- 0. PERFORMANCE SETUP ---
try:
    from numba import jit
    JIT_AVAILABLE = True
    print("[+] Numba JIT detected: High-Performance Mode ACTIVE.")
except ImportError:
    def jit(nopython=True):
        def decorator(func):
            return func
        return decorator
    JIT_AVAILABLE = False
    print("[!] Numba not found: Running in pure Python (Slow). Install 'numba' for speed.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("GridEngine_v9")

# ==========================================
# 1. STRICT DATA MODELS
# ==========================================


class ConnectivityNode(BaseModel):
    id: str
    voltage_level_kv: float = Field(..., gt=0)
    type: Literal['PQ', 'PV', 'SLACK'] = 'PQ'


class Terminal(BaseModel):
    id: str
    connectivity_node: str
    conducting_equipment: str


class PerLengthPhaseImpedance(BaseModel):
    id: str
    r_ohm_per_km: float = Field(..., ge=0)
    x_ohm_per_km: float = Field(..., ge=0)


class ConductingEquipment(BaseModel):
    id: str


class ACLineSegment(ConductingEquipment):
    length_km: float = Field(..., gt=0)
    impedance_ref: str

    def get_z_total(self, catalog: Dict[str,
                    PerLengthPhaseImpedance]) -> complex:
        imp = catalog[self.impedance_ref]
        return complex(imp.r_ohm_per_km, imp.x_ohm_per_km) * self.length_km


class EnergyConsumer(ConductingEquipment):
    p_mw: float
    q_mvar: float


class GridTopology(BaseModel):
    nodes: List[ConnectivityNode]
    terminals: List[Terminal]
    segments: List[ACLineSegment]
    consumers: List[EnergyConsumer] = []
    impedance_catalog: Dict[str, PerLengthPhaseImpedance] = {}

    @model_validator(mode='after')
    def validate_connectivity(self):
        node_ids = {n.id for n in self.nodes}
        equip_ids = {e.id for e in self.segments + self.consumers}

        for t in self.terminals:
            if t.connectivity_node not in node_ids:
                raise ValueError(f"[!] Invalid Node Ref on Terminal: {t.id}")
            if t.conducting_equipment not in equip_ids:
                raise ValueError(f"[!] Invalid Equipment Ref on Terminal: {t.id}")

        return self

# ==========================================
# 2. TOPOLOGY GENERATOR (Realistic Params)
# ==========================================


def generate_linear_feeder(
        n_nodes: int,
        voltage_kv: float = 20.0) -> GridTopology:
    """
    Generates a realistic suburban feeder.
    Spacing: 0.6 km (Standard node/substation spacing)
    Cable: NA2XS2Y 1x150 mm² (R=0.20, X=0.12)
    """

    # Standard Impedance Catalog
    na2xs2y_150 = PerLengthPhaseImpedance(
        id="cable_na2xs2y_1x150", r_ohm_per_km=0.20, x_ohm_per_km=0.12
    )
    catalog = {na2xs2y_150.id: na2xs2y_150}

    nodes = [
        ConnectivityNode(
            id="source",
            voltage_level_kv=voltage_kv,
            type='SLACK')]
    segments = []
    terminals = []

    prev_node_id = "source"
    for i in range(1, n_nodes + 1):
        curr_node_id = f"node_{i}"
        line_id = f"segment_{i}"

        nodes.append(
            ConnectivityNode(
                id=curr_node_id,
                voltage_level_kv=voltage_kv))

        # The Equipment
        segments.append(ACLineSegment(
            id=line_id,
            length_km=0.6,
            impedance_ref=na2xs2y_150.id
        ))

        # The Linkage via Terminals
        terminals.append(
            Terminal(
                id=f"t_{line_id}_1",
                connectivity_node=prev_node_id,
                conducting_equipment=line_id))
        terminals.append(
            Terminal(
                id=f"t_{line_id}_2",
                connectivity_node=curr_node_id,
                conducting_equipment=line_id))

        prev_node_id = curr_node_id

    return GridTopology(
        nodes=nodes,
        terminals=terminals,
        segments=segments,
        impedance_catalog=catalog)

# ==========================================
# 3. HIGH-PERFORMANCE PHYSICS KERNEL
# ==========================================


@jit(nopython=True)
def _fast_fbs_kernel(v_mag: np.ndarray,
                     v_complex: np.ndarray,
                     p_inj: np.ndarray,
                     q_inj: np.ndarray,
                     z_matrix: np.ndarray,
                     parent_indices: np.ndarray,
                     order_indices: np.ndarray,
                     tol: float,
                     max_iter: int) -> Tuple[np.ndarray,
                                             bool]:
    n_nodes = len(v_complex)
    converged = False

    for _ in range(max_iter):
        max_err = 0.0
        i_inj = (p_inj - 1j * q_inj) / (v_complex + 1e-9)

        # Backward Sweep
        i_branch = np.zeros(n_nodes, dtype=np.complex128)
        for k in range(n_nodes - 1, 0, -1):
            idx = order_indices[k]
            i_branch[idx] += i_inj[idx]
            parent = parent_indices[idx]
            if parent != -1:
                i_branch[parent] += i_branch[idx]

        # Forward Sweep
        for k in range(1, n_nodes):
            idx = order_indices[k]
            parent = parent_indices[idx]
            z = z_matrix[idx]
            v_new = v_complex[parent] - (i_branch[idx] * z)

            err = np.abs(v_new - v_complex[idx])
            if err > max_err:
                max_err = err
            v_complex[idx] = v_new
            v_mag[idx] = np.abs(v_new)

        if max_err < tol:
            converged = True
            break

    return v_complex, converged


class FastGridEngine:
    def __init__(self, grid: GridTopology):
        self.grid = grid
        self._compile_topology()

    def _compile_topology(self):
        # 1. Index Mapping
        self.node_map = {n.id: i for i, n in enumerate(self.grid.nodes)}

        # --- FIX IS HERE ---
        # We just swap the key (node_id) and value (i)
        self.idx_map = {i: node_id for node_id, i in self.node_map.items()}

        n = len(self.grid.nodes)

        # 2. Graph Analysis (Bridging via Terminals)
        G = nx.Graph()

        # Build equipment-to-nodes map
        equip_nodes_map = {}
        for t in self.grid.terminals:
            if t.conducting_equipment not in equip_nodes_map:
                equip_nodes_map[t.conducting_equipment] = []
            equip_nodes_map[t.conducting_equipment].append(t.connectivity_node)

        for segment in self.grid.segments:
            nodes = equip_nodes_map.get(segment.id, [])
            if len(nodes) == 2:
                z = segment.get_z_total(self.grid.impedance_catalog)
                G.add_edge(nodes[0], nodes[1], z=z)

        root_id = self.grid.nodes[0].id
        bfs_tree = nx.bfs_tree(G, source=root_id)
        self.order = [self.node_map[node] for node in bfs_tree]

        self.z_array = np.zeros(n, dtype=np.complex128)
        self.parents = np.full(n, -1, dtype=np.int32)
        self.base_kv = np.array([n.voltage_level_kv for n in self.grid.nodes])

        for u, v in bfs_tree.edges():
            idx_u, idx_v = self.node_map[u], self.node_map[v]
            self.parents[idx_v] = idx_u
            self.z_array[idx_v] = G.edges[u, v]['z']

    def solve(
            self, active_consumers: List[EnergyConsumer]) -> Dict[str, complex]:
        n = len(self.grid.nodes)
        p_inj = np.zeros(n)
        q_inj = np.zeros(n)

        for consumer in active_consumers:
            # Find the terminal for this consumer
            consumer_node = None
            for t in self.grid.terminals:
                if t.conducting_equipment == consumer.id:
                    consumer_node = t.connectivity_node
                    break

            if consumer_node and consumer_node in self.node_map:
                idx = self.node_map[consumer_node]
                p_inj[idx] += consumer.p_mw
                q_inj[idx] += consumer.q_mvar

        v_complex = self.base_kv.astype(np.complex128)

        # Run Kernel
        v_res, conv = _fast_fbs_kernel(
            np.abs(v_complex), v_complex, p_inj, q_inj,
            self.z_array, self.parents, np.array(self.order),
            tol=1e-5, max_iter=20
        )
        if not conv:
            logger.warning("[!] Solver divergence detected")
        return {self.idx_map[i]: v_res[i] for i in range(n)}

# ==========================================
# 4. VECTORIZED CONTROLLER
# ==========================================


class VectorizedController:
    def compute_batch_q(self, v_pu_array: np.ndarray) -> np.ndarray:
        """
        Calculates Reactive Power (Q) injection to fix voltage.
        """
        err = 0.94 - v_pu_array  # Target 0.94 p.u.
        q_out = np.zeros_like(err)

        # Inject Q if Voltage < 0.94
        low_mask = (err > 0.0)
        q_out[low_mask] = err[low_mask] * 10.0  # Gain Factor
        return q_out


# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n=== GRID ENGINE: REALISTIC VALIDATION ===\n")

    # A. SETUP (Realistic Suburban Feeder)
    #
    N_NODES = 20
    grid = generate_linear_feeder(N_NODES, voltage_kv=20.0)

    # 2.5 MW is a realistic heavy load (e.g., small factory or EV park)
    end_node_id = grid.nodes[-1].id
    heavy_load = EnergyConsumer(id="heavy_load", p_mw=2.5, q_mvar=0.5)

    # Link load to node via Terminal
    load_terminal = Terminal(
        id="t_heavy_load_1",
        connectivity_node=end_node_id,
        conducting_equipment=heavy_load.id)

    grid.consumers.append(heavy_load)
    grid.terminals.append(load_terminal)

    # B. SOLVE
    t0 = time.perf_counter()
    engine = FastGridEngine(grid)
    res = engine.solve(grid.consumers)
    solve_ms = (time.perf_counter() - t0) * 1000

    # C. ANALYZE
    v_end = abs(res[end_node_id]) / 20.0
    print(f"[*] Solved in {solve_ms:.3f} ms")
    print(f"[*] End Voltage: {v_end:.4f} p.u.")

    if v_end < 0.90:
        print("[!] CRITICAL: Voltage Unsafe!")
        # Optional: Activate Controller here
    else:
        print("[+] SAFE: Grid Operating within Limits.")

    # D. VISUALIZE
    #
    dist_km = [i * 0.6 for i in range(len(grid.nodes))]  # 0.6km segments
    v_profile = [abs(res[n.id]) / 20.0 for n in grid.nodes]

    plt.figure(figsize=(10, 5))
    plt.plot(dist_km, v_profile, 'o-', color='navy', label='Voltage Profile')
    plt.axhline(0.9, color='red', linestyle='--', label='Min Limit (0.9 p.u.)')
    plt.fill_between(
        dist_km,
        0.9,
        1.05,
        color='green',
        alpha=0.1,
        label='Safe Zone')
    plt.xlabel('Distance (km)')
    plt.ylabel('Voltage (p.u.)')
    plt.title(f'Feeder Analysis: {N_NODES} Nodes / 2.5 MW Load')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show() # commented out for test running without hanging
