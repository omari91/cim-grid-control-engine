import numpy as np
import networkx as nx
import unittest
import matplotlib.pyplot as plt
import requests
import logging
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Literal, Optional, Dict
from scipy.sparse import csr_matrix

# ==========================================
# 0. CONFIGURATION & CONSTANTS
# ==========================================
GLOBAL_TRAFO_LIMIT_MW = 45.0
# API Config (In production, load from os.environ)
NTP_CONFIG = {
    "base_url": "https://ds.netztransparenz.de/api/v1/data",
    # Placeholder credentials
    "client_id": "demo_id", "client_secret": "demo_secret", "token_url": "https://demo.url"
}
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. LIVE DATA INGESTION LAYER
# ==========================================
class NTPClient:
    """
    Handles authentication and data fetching from Netztransparenz Platform.
    """
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.token = None
        self.session = requests.Session()

    def authenticate(self):
        # Placeholder for OAuth2 logic
        self.token = "simulated_token"

    def fetch_latest_generation(self, control_area="50Hertz") -> float:
        """
        Fetches live generation data.
        """
        if not self.token: self.authenticate()

        # Use timezone-aware datetime to avoid DeprecationWarning
        now = datetime.now(timezone.utc)
        
        # SIMULATION MODE: 
        # Simulating a live API response for demonstration purposes.
        simulated_p_mw = np.random.normal(25.0, 5.0) 
        return max(0.0, simulated_p_mw)

# ==========================================
# 2. DOMAIN DATA LAYER (Pydantic / CIM)
# ==========================================
class Bus(BaseModel):
    id: str
    voltage_level_kv: float = Field(..., gt=0)
    type: Literal['PQ', 'PV', 'SLACK'] = 'PQ'

class ACLineSegment(BaseModel):
    id: str
    from_bus: str
    to_bus: str
    length_km: float = Field(..., gt=0)
    r_ohm_per_km: float = Field(..., ge=0)
    x_ohm_per_km: float = Field(..., ge=0)
    c_nf_per_km: float = Field(default=0.0, ge=0)
    max_current_a: float = Field(..., gt=0)

    @property
    def total_impedance(self) -> complex:
        return complex(self.r_ohm_per_km, self.x_ohm_per_km) * self.length_km

class Load(BaseModel):
    id: str
    bus: str
    p_mw: float
    q_mvar: float

class GridModel(BaseModel):
    buses: List[Bus]
    lines: List[ACLineSegment]
    loads: List[Load] = []

    @model_validator(mode='after')
    def check_topology_integrity(self):
        bus_ids = {b.id for b in self.buses}
        for line in self.lines:
            if line.from_bus not in bus_ids or line.to_bus not in bus_ids:
                raise ValueError(f"Line {line.id} has disconnected terminals.")
        return self
    
    def get_bus(self, bus_id: str) -> Bus:
        return next(b for b in self.buses if b.id == bus_id)

# ==========================================
# 3. TOPOLOGY & SOLVER CORE
# ==========================================
class TopologyService:
    @staticmethod
    def build_graph(grid: GridModel) -> nx.Graph:
        G = nx.Graph()
        for bus in grid.buses: G.add_node(bus.id, obj=bus)
        for line in grid.lines:
            G.add_edge(line.from_bus, line.to_bus, id=line.id, obj=line)
        return G

    @staticmethod
    def get_bfs_ordering(G: nx.Graph, root_id: str) -> List[str]:
        return list(nx.topological_sort(nx.bfs_tree(G, source=root_id)))

class PowerFlowSolver:
    """
    Forward-Backward Sweep (FBS) Solver for Radial Distribution Grids.
    
    ENGINEERING ASSUMPTIONS:
    1. Balanced Three-Phase System: We model the positive sequence only.
    2. Power Values: Interpreted as total three-phase power (MW).
    3. Voltage: Interpreted as line-to-line magnitude (kV).
    """
    
    @staticmethod
    def solve_fbs(grid: GridModel, root_id: str, tol=1e-4) -> Dict[str, complex]:
        G = TopologyService.build_graph(grid)
        order = TopologyService.get_bfs_ordering(G, root_id)
        
        voltages = {b.id: complex(b.voltage_level_kv, 0) for b in grid.buses}
        
        for _ in range(20): # Max Iterations
            max_err = 0.0
            node_currents = {n: 0j for n in order}
            
            #1. Calculate Injections (I = S*/V*) 
            for load in grid.loads:
                v = voltages[load.bus]
                if abs(v) > 1e-5:
                    s = complex(load.p_mw, load.q_mvar)
                    node_currents[load.bus] += (s / v).conjugate()
            
            # 2. Sweep
            for node in order:
                if node == root_id: continue
                
                # TODO: Optimization - Store parent pointers during BFS 
                # to avoid O(N^2) lookups in this loop for large grids.
                neighbors = list(G.neighbors(node))
                parent = next(n for n in neighbors if order.index(n) < order.index(node))
                
                line = G.edges[parent, node]['obj']
                z = line.total_impedance
                
                v_new = voltages[parent] - (node_currents[node] * z)
                max_err = max(max_err, abs(v_new - voltages[node]))
                voltages[node] = v_new
                
            if max_err < tol: return voltages
        return voltages

class FuzzyVoltVarController:
    """
    Implements Fuzzy Logic for Volt-VAR Optimization.
    
    NOTE: This is a conceptual controller for demonstration purposes.
    Real inverters would require specific ramp-rate limits and deadbands per grid codes (e.g., VDE-AR-N 4105).
    """
    def __init__(self): pass

    def _membership_triangle(self, x: float, left: float, center: float, right: float) -> float:
        if x <= left or x >= right: return 0.0
        elif x == center: return 1.0
        elif x < center: return (x - left) / (center - left)
        else: return (right - x) / (right - center)

    def compute_q_setpoint(self, v_measured_pu: float) -> float:
        # Simplified Fuzzy Inference System
        error = 1.0 - v_measured_pu # Error > 0 means voltage is low
        
        # Fuzzify
        mu_NB = self._membership_triangle(error, -0.15, -0.10, -0.0) # Voltage High
        mu_Z  = self._membership_triangle(error, -0.05,  0.00,  0.05) # Nominal
        mu_PB = self._membership_triangle(error,  0.00,  0.10,  0.15) # Voltage Low
        
        # Defuzzify (Center of Gravity)
        num = (mu_NB * -1.0) + (mu_Z * 0.0) + (mu_PB * 1.0)
        den = mu_NB + mu_Z + mu_PB
        return num / den if den != 0 else 0.0

# ==========================================
# 4. ANALYTICS & CONSTRAINTS ENGINE
# ==========================================
class AnalyticsEngine:
    @staticmethod
    def check_constraints(grid: GridModel, results: Dict[str, complex]) -> bool:
        # 1. Voltage Check
        for bus_id, v_complex in results.items():
            base_kv = grid.get_bus(bus_id).voltage_level_kv
            v_pu = abs(v_complex) / base_kv
            if not (0.9 <= v_pu <= 1.1): return False

        # 2. Global Trafo Limit
        total_p_mw = sum(l.p_mw for l in grid.loads)
        if abs(total_p_mw) > GLOBAL_TRAFO_LIMIT_MW:
             return False
        return True

    @staticmethod
    def hosting_capacity(grid: GridModel, target_node: str) -> float:
        """
        Calculates max generation capacity using Binary Search O(log N).
        
        NOTE: This implementation mutates the grid state for performance.
        In a multi-threaded production environment, use deepcopy() or immutable data structures.
        """
        min_p, max_p = 0.0, 100.0
        
        while (max_p - min_p) > 0.05:
            mid_p = (min_p + max_p) / 2
            # Inject Test Gen (Negative Load)
            grid.loads.append(Load(id="test_gen", bus=target_node, p_mw=-mid_p, q_mvar=0))
            
            try:
                res = PowerFlowSolver.solve_fbs(grid, "source_bus")
                if AnalyticsEngine.check_constraints(grid, res):
                    min_p = mid_p 
                else:
                    max_p = mid_p 
            except:
                max_p = mid_p 
                
            grid.loads.pop() # Restore state
        return min_p

# ==========================================
# 5. VISUALIZATION & MAIN
# ==========================================
def plot_results(base_res, hc_res, hc_mw):
    dists = [0.0, 5.0]
    v_base = [abs(base_res[n])/10.0 for n in ["source_bus", "load_bus"]]
    v_hc = [abs(hc_res[n])/10.0 for n in ["source_bus", "load_bus"]]
    
    plt.figure(figsize=(8, 5))
    plt.plot(dists, v_base, 'o--', label='Live Load Scenario')
    plt.plot(dists, v_hc, 's-', label=f'With +{hc_mw:.1f}MW Gen')
    plt.axhline(1.1, c='r', ls=':', label='Limit')
    plt.axhline(0.9, c='r', ls=':')
    plt.title("Voltage Profile: Live Data vs Hosting Capacity")
    plt.legend(); plt.grid(True, alpha=0.3); plt.show()

if __name__ == "__main__":
    print("--- ENVELIO GRID ENGINE: LIVE PRODUCTION MODE ---\n")
    
    # 1. Ingest
    ntp_client = NTPClient(NTP_CONFIG)
    live_load_mw = ntp_client.fetch_latest_generation()
    print(f"[*] Ingested Live Load Data: {live_load_mw:.2f} MW")
    
    # 2. Model
    grid = GridModel(
        buses=[Bus(id="source_bus", voltage_level_kv=10.0, type='SLACK'), Bus(id="load_bus", voltage_level_kv=10.0)],
        lines=[ACLineSegment(id="L1", from_bus="source_bus", to_bus="load_bus", length_km=5.0, r_ohm_per_km=0.1, x_ohm_per_km=0.1, max_current_a=400)]
    )
    grid.loads = [Load(id="live_load", bus="load_bus", p_mw=live_load_mw, q_mvar=live_load_mw*0.2)]
    
    # 3. Solve & Analyze
    base_res = PowerFlowSolver.solve_fbs(grid, "source_bus")
    print(f"[*] Base Voltage: {abs(base_res['load_bus'])/10.0:.4f} p.u.")
    
    hc_mw = AnalyticsEngine.hosting_capacity(grid, "load_bus")
    print(f"[*] Hosting Capacity: {hc_mw:.2f} MW")
    
    if hc_mw > GLOBAL_TRAFO_LIMIT_MW:
        print(f"[!] Warning: Limited by Trafo Rating ({GLOBAL_TRAFO_LIMIT_MW} MW)")

    # 4. Fuzzy Control Demo
    # Simulate high voltage scenario
    fvc = FuzzyVoltVarController()
    q_out = fvc.compute_q_setpoint(1.08) # 1.08 p.u. input
    print(f"[*] Fuzzy Control Action for 1.08 p.u.: Absorb {abs(q_out):.2f} MVar")

    # 5. Visualize
    grid.loads.append(Load(id="max_gen", bus="load_bus", p_mw=-hc_mw, q_mvar=0))
    hc_res = PowerFlowSolver.solve_fbs(grid, "source_bus")
    plot_results(base_res, hc_res, hc_mw)
