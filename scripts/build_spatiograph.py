import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sumolib.net import readNet


@dataclass(frozen=True)
class DirectedRoad:
    src: str
    dst: str
    edge_id: str
    num_lanes: int
    length_m: float
    speed_mps: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 1: Build a directed spatio-graph G=(V,E) from a SUMO .net.xml. "
            "V are signalized junctions (traffic lights). E are directed road segments connecting them."
        )
    )
    parser.add_argument(
        "--net",
        type=str,
        default="scenarios/grid_3x3_lanes3_dynamic_dense/grid_3x3_lanes3_dynamic_dense.net.xml",
        help="Path to the SUMO .net.xml file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output JSON path. If omitted, prints a summary only.",
    )
    parser.add_argument(
        "--include-internal",
        action="store_true",
        help="Include internal edges (normally should be False for GNN road graph).",
    )
    return parser.parse_args()


def _is_internal_edge(edge_id: str) -> bool:
    # SUMO internal edges usually start with ':'
    return edge_id.startswith(":")


def build_graph(net_path: Path, include_internal: bool = False) -> dict[str, Any]:
    net = readNet(str(net_path))

    # Control nodes: all traffic light IDs in the net.
    # In grid nets, tls ids are usually like 'gneJ0' etc; in RiLSA example it's '0'.
    tls_ids = sorted(tls.getID() for tls in net.getTrafficLights())
    control_nodes = set(tls_ids)

    # Map junction (node) id -> whether it's controlled by a tls.
    # sumolib exposes tls ids, and each tls controls a node with same id in typical nets.
    # We'll still guard with existence in net nodes.
    node_ids = {n.getID() for n in net.getNodes()}
    control_nodes = {tls for tls in control_nodes if tls in node_ids or tls in tls_ids}

    # Incoming lanes per control node (useful for Step 2).
    incoming_lanes: dict[str, list[str]] = {nid: [] for nid in sorted(control_nodes)}

    edges: list[DirectedRoad] = []

    for edge in net.getEdges():
        edge_id = edge.getID()
        if not include_internal and _is_internal_edge(edge_id):
            continue

        from_node = edge.getFromNode().getID()
        to_node = edge.getToNode().getID()

        # We only keep graph edges that connect two control nodes.
        if from_node in control_nodes and to_node in control_nodes:
            edges.append(
                DirectedRoad(
                    src=from_node,
                    dst=to_node,
                    edge_id=edge_id,
                    num_lanes=edge.getLaneNumber(),
                    length_m=float(edge.getLength()),
                    speed_mps=float(edge.getSpeed()),
                )
            )

        # For step 2, collect lanes that go INTO a control node.
        if to_node in control_nodes:
            for lane in edge.getLanes():
                incoming_lanes[to_node].append(lane.getID())

    # Deduplicate lane ids while keeping stable order
    for nid, lanes in incoming_lanes.items():
        seen: set[str] = set()
        deduped: list[str] = []
        for lane_id in lanes:
            if lane_id in seen:
                continue
            seen.add(lane_id)
            deduped.append(lane_id)
        incoming_lanes[nid] = deduped

    graph = {
        "net": str(net_path.as_posix()),
        "control_nodes": sorted(control_nodes),
        "edges": [asdict(e) for e in edges],
        "incoming_lanes": incoming_lanes,
        "num_control_nodes": len(control_nodes),
        "num_directed_edges": len(edges),
    }
    return graph


def main() -> int:
    args = _parse_args()
    net_path = Path(args.net)
    if not net_path.exists():
        raise SystemExit(f".net.xml not found: {net_path}")

    graph = build_graph(net_path, include_internal=args.include_internal)

    print("Net:", graph["net"])
    print("Control nodes (TLS junctions):", graph["num_control_nodes"])
    print("Directed edges (between TLS junctions):", graph["num_directed_edges"])

    # Show a small preview
    cn = graph["control_nodes"]
    preview_nodes = cn[: min(5, len(cn))]
    print("Preview control nodes:", preview_nodes)

    # Count incoming lanes stats
    incoming_counts = [len(graph["incoming_lanes"][nid]) for nid in cn]
    if incoming_counts:
        print(
            "Incoming lanes per TLS (min/avg/max):",
            min(incoming_counts),
            round(sum(incoming_counts) / len(incoming_counts), 2),
            max(incoming_counts),
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Wrote graph JSON:", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
