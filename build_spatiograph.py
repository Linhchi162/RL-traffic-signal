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
    return edge_id.startswith(":")


def _is_internal_edge_obj(edge: Any) -> bool:
    try:
        return edge.getFunction() == "internal"
    except Exception:
        return False


def build_graph(net_path: Path, include_internal: bool = False) -> dict[str, Any]:
    net = readNet(str(net_path))

    # Step 1A: classify nodes (V)
    control_nodes: list[str] = []
    non_control_nodes: list[str] = []
    for node in net.getNodes():
        node_id = node.getID()
        if node.getType() == "traffic_light":
            control_nodes.append(node_id)
        else:
            non_control_nodes.append(node_id)

    control_set = set(control_nodes)

    incoming_lanes: dict[str, list[str]] = {nid: [] for nid in sorted(control_set)}

    edges: list[DirectedRoad] = []
    edges_data: list[dict[str, Any]] = []

    for edge in net.getEdges():
        edge_id = edge.getID()
        if not include_internal and (_is_internal_edge(edge_id) or _is_internal_edge_obj(edge)):
            continue

        from_node = edge.getFromNode().getID()
        to_node = edge.getToNode().getID()

        edges_data.append(
            {
                "edge_id": edge_id,
                "send_k": from_node,
                "rec_k": to_node,
                "c_k": int(edge.getLaneNumber()),
                "length_m": float(edge.getLength()),
                "speed_mps": float(edge.getSpeed()),
            }
        )

        if from_node in control_set and to_node in control_set:
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

        if to_node in control_set:
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
        "non_control_nodes": sorted(non_control_nodes),
        "edges_data": edges_data,
        "edges_between_control_nodes": [asdict(e) for e in edges],
        "incoming_lanes": incoming_lanes,
        "num_control_nodes": len(control_nodes),
        "num_non_control_nodes": len(non_control_nodes),
        "num_directed_edges": len(edges_data),
        "num_directed_edges_between_control_nodes": len(edges),
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
    print("Non-control nodes:", graph["num_non_control_nodes"])
    print("Directed edges (all, non-internal):", graph["num_directed_edges"])
    print(
        "Directed edges (between TLS junctions):",
        graph["num_directed_edges_between_control_nodes"],
    )

    cn = graph["control_nodes"]
    preview_nodes = cn[: min(5, len(cn))]
    print("Preview control nodes:", preview_nodes)

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
