import xml.etree.ElementTree as ET
import time
from typing import List

from main import ConnectivityNode, ACLineSegment, EnergyConsumer, GridTopology, FastGridEngine, Terminal, PerLengthPhaseImpedance

CIM_NS = {'cim': 'http://iec.ch/TC57/2013/CIM-schema-cim16#',
          'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'}


def parse_cim_xml(file_path: str) -> GridTopology:
    """
    Parses a CIM XML (RDF) file and returns a validated GridTopology object.
    Translating basic XML elements into the new strict Terminal linkages and Impedance properties.
    """
    print(f"[*] Parsing CIM XML file: {file_path}")
    tree = ET.parse(file_path)
    root = tree.getroot()

    nodes: List[ConnectivityNode] = []
    terminals: List[Terminal] = []
    segments: List[ACLineSegment] = []
    consumers: List[EnergyConsumer] = []
    impedance_catalog = {}

    # 1. Parse Connectivity Nodes
    for node_elem in root.findall('cim:ConnectivityNode', CIM_NS):
        node_id = node_elem.attrib.get(f"{{{CIM_NS['rdf']}}}ID")
        voltage_elem = node_elem.find('cim:ConnectivityNode.voltageLevelKV', CIM_NS)
        v_kv = float(voltage_elem.text) if voltage_elem is not None else 20.0
        type_elem = node_elem.find('cim:ConnectivityNode.type', CIM_NS)
        n_type = type_elem.text if type_elem is not None else 'PQ'

        nodes.append(ConnectivityNode(id=node_id, voltage_level_kv=v_kv, type=n_type))

    # 2. Parse ACLineSegments
    for line_elem in root.findall('cim:ACLineSegment', CIM_NS):
        line_id = line_elem.attrib.get(f"{{{CIM_NS['rdf']}}}ID")

        from_node = line_elem.find('cim:ACLineSegment.fromNode', CIM_NS).attrib.get(
            f"{{{CIM_NS['rdf']}}}resource").strip('#')
        to_node = line_elem.find('cim:ACLineSegment.toNode', CIM_NS).attrib.get(
            f"{{{CIM_NS['rdf']}}}resource").strip('#')

        length_km = float(line_elem.find('cim:ACLineSegment.length', CIM_NS).text)
        r = float(line_elem.find('cim:ACLineSegment.r', CIM_NS).text)
        x = float(line_elem.find('cim:ACLineSegment.x', CIM_NS).text)

        # Map to catalog pattern
        imp_id = f"imp_{line_id}"
        impedance_catalog[imp_id] = PerLengthPhaseImpedance(id=imp_id, r_ohm_per_km=r, x_ohm_per_km=x)

        segments.append(ACLineSegment(
            id=line_id,
            length_km=length_km,
            impedance_ref=imp_id
        ))

        # Build linkages via Terminals
        terminals.append(Terminal(id=f"t_{line_id}_1", connectivity_node=from_node, conducting_equipment=line_id))
        terminals.append(Terminal(id=f"t_{line_id}_2", connectivity_node=to_node, conducting_equipment=line_id))

    # 3. Parse EnergyConsumers
    for load_elem in root.findall('cim:EnergyConsumer', CIM_NS):
        load_id = load_elem.attrib.get(f"{{{CIM_NS['rdf']}}}ID")

        node_ref = load_elem.find('cim:EnergyConsumer.node', CIM_NS).attrib.get(
            f"{{{CIM_NS['rdf']}}}resource").strip('#')
        p = float(load_elem.find('cim:EnergyConsumer.p', CIM_NS).text)
        q = float(load_elem.find('cim:EnergyConsumer.q', CIM_NS).text)

        consumers.append(EnergyConsumer(
            id=load_id,
            p_mw=p,
            q_mvar=q
        ))

        # Terminal for consumer
        terminals.append(Terminal(id=f"t_{load_id}_1", connectivity_node=node_ref, conducting_equipment=load_id))

    print(f"[+] Loaded {len(nodes)} Nodes, {len(terminals)} Terminals, {len(segments)} Lines, {len(consumers)} Consumers.")
    return GridTopology(nodes=nodes, terminals=terminals, segments=segments, consumers=consumers, impedance_catalog=impedance_catalog)


if __name__ == "__main__":
    print("\n=== XML INGESTION TEST ===")

    # 1. Parse File
    grid_topology = parse_cim_xml("sample_grid.xml")

    # 2. Extract Consumers (Loads) for the solve step
    consumers = grid_topology.consumers

    # 3. Solve using Engine
    t0 = time.perf_counter()
    engine = FastGridEngine(grid_topology)
    res = engine.solve(consumers)
    solve_ms = (time.perf_counter() - t0) * 1000

    # 4. Display Results
    print(f"[*] Engine Solved in {solve_ms:.3f} ms")

    for node_id, v_complex in res.items():
        v_pu = abs(v_complex) / 20.0
        print(f"    - {node_id}: {v_pu:.4f} p.u.")
