"""
build_net.py -- Rebuild caliberated_net.xml using netconvert.

Vietnam-style intersection:
  - 4 approaches, 2 mixed lanes each:
      Lane 0 (outer/curb): straight + right turn
      Lane 1 (inner): straight + left turn
  - traffic_light_right_on_red: right turns always permissive ('s')
  - 2 main phases: NS green / WE green (+ 2 yellow transitions)

Usage:
    cd nets && python build_net.py
"""

import subprocess
import sys
import os
from pathlib import Path
from xml.etree import ElementTree as ET

HERE = Path(__file__).parent

NODES_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<nodes>
    <node id="t" x="150" y="150" type="traffic_light_right_on_red"/>
    <node id="n" x="150" y="300" type="dead_end"/>
    <node id="s" x="150" y="0"   type="dead_end"/>
    <node id="e" x="300" y="150" type="dead_end"/>
    <node id="w" x="0"   y="150" type="dead_end"/>
</nodes>
"""

EDGES_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<edges>
    <edge id="n_t" from="n" to="t" numLanes="3" speed="13.89" spreadType="right"/>
    <edge id="t_n" from="t" to="n" numLanes="3" speed="13.89" spreadType="right"/>
    <edge id="s_t" from="s" to="t" numLanes="3" speed="13.89" spreadType="right"/>
    <edge id="t_s" from="t" to="s" numLanes="3" speed="13.89" spreadType="right"/>
    <edge id="e_t" from="e" to="t" numLanes="3" speed="13.89" spreadType="right"/>
    <edge id="t_e" from="t" to="e" numLanes="3" speed="13.89" spreadType="right"/>
    <edge id="w_t" from="w" to="t" numLanes="3" speed="13.89" spreadType="right"/>
    <edge id="t_w" from="t" to="w" numLanes="3" speed="13.89" spreadType="right"/>
</edges>
"""

# Lane 0 (outer/curb):  straight + right turn
# Lane 1 (middle):      straight only
# Lane 2 (inner/left):  straight + left turn
#
# Directions (right-hand traffic):
#   s_t (from South, heading North): straight=t_n, right=t_e, left=t_w
#   n_t (from North, heading South): straight=t_s, right=t_w, left=t_e
#   e_t (from East,  heading West):  straight=t_w, right=t_n, left=t_s
#   w_t (from West,  heading East):  straight=t_e, right=t_s, left=t_n
CONNECTIONS_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<connections>
    <!-- South approach (heading North) -->
    <connection from="s_t" to="t_n" fromLane="0" toLane="0"/>
    <connection from="s_t" to="t_e" fromLane="0" toLane="0"/>
    <connection from="s_t" to="t_n" fromLane="1" toLane="1"/>
    <connection from="s_t" to="t_n" fromLane="2" toLane="2"/>
    <connection from="s_t" to="t_w" fromLane="2" toLane="0"/>

    <!-- North approach (heading South) -->
    <connection from="n_t" to="t_s" fromLane="0" toLane="0"/>
    <connection from="n_t" to="t_w" fromLane="0" toLane="0"/>
    <connection from="n_t" to="t_s" fromLane="1" toLane="1"/>
    <connection from="n_t" to="t_s" fromLane="2" toLane="2"/>
    <connection from="n_t" to="t_e" fromLane="2" toLane="0"/>

    <!-- East approach (heading West) -->
    <connection from="e_t" to="t_w" fromLane="0" toLane="0"/>
    <connection from="e_t" to="t_n" fromLane="0" toLane="0"/>
    <connection from="e_t" to="t_w" fromLane="1" toLane="1"/>
    <connection from="e_t" to="t_w" fromLane="2" toLane="2"/>
    <connection from="e_t" to="t_s" fromLane="2" toLane="0"/>

    <!-- West approach (heading East) -->
    <connection from="w_t" to="t_e" fromLane="0" toLane="0"/>
    <connection from="w_t" to="t_s" fromLane="0" toLane="0"/>
    <connection from="w_t" to="t_e" fromLane="1" toLane="1"/>
    <connection from="w_t" to="t_e" fromLane="2" toLane="2"/>
    <connection from="w_t" to="t_n" fromLane="2" toLane="0"/>
</connections>
"""

NS_EDGES = {"s_t", "n_t"}
WE_EDGES = {"e_t", "w_t"}


def patch_tls(net_path: Path) -> None:
    """Replace the generated TLS phases with Vietnam-style NS/WE + right-on-red."""
    tree = ET.parse(net_path)
    root = tree.getroot()

    link_info: dict[int, dict] = {}
    for conn in root.iter("connection"):
        if conn.get("tl") is None or conn.get("linkIndex") is None:
            continue
        idx = int(conn.get("linkIndex"))
        link_info[idx] = {
            "from_edge": conn.get("from", ""),
            "dir": conn.get("dir", "s"),
        }

    if not link_info:
        print("[patch_tls] No TLS-controlled connections found -- skipping.")
        return

    n = max(link_info) + 1
    print(f"[patch_tls] {n} controlled links")

    def build_state(ns_active: bool, yellow: bool) -> str:
        chars = []
        for i in range(n):
            info = link_info.get(i, {"from_edge": "", "dir": "s"})
            d    = info["dir"]
            edge = info["from_edge"]

            if d == "r":           # right turn: always permissive
                chars.append("s")
                continue

            is_ns = edge in NS_EDGES
            is_we = edge in WE_EDGES
            active = (ns_active and is_ns) or (not ns_active and is_we)

            if yellow:
                chars.append("y" if active else "r")
            else:
                chars.append("G" if active else "r")
        return "".join(chars)

    states = {
        "ns_green":  build_state(True,  False),
        "ns_yellow": build_state(True,  True),
        "we_green":  build_state(False, False),
        "we_yellow": build_state(False, True),
    }
    for name, s in states.items():
        print(f"  {name:10s}: {s}")

    for tl_logic in root.iter("tlLogic"):
        if tl_logic.get("id") == "t":
            for ph in list(tl_logic):
                tl_logic.remove(ph)
            for dur, state in [(40, states["ns_green"]),
                               (4,  states["ns_yellow"]),
                               (40, states["we_green"]),
                               (4,  states["we_yellow"])]:
                el = ET.SubElement(tl_logic, "phase")
                el.set("duration", str(dur))
                el.set("state", state)
            break

    ET.indent(root, space="    ")
    tree.write(net_path, encoding="unicode", xml_declaration=True)
    print(f"[patch_tls] Written: {net_path.name}")


def main():
    sumo_home = os.environ.get("SUMO_HOME", r"D:\Program Files\Eclipse\Sumo")
    netconvert = Path(sumo_home) / "bin" / "netconvert.exe"
    if not netconvert.exists():
        netconvert = Path("netconvert")

    nodes_f = HERE / "_nodes.xml"
    edges_f = HERE / "_edges.xml"
    conns_f = HERE / "_connections.xml"
    out_f   = HERE / "caliberated_net.xml"

    nodes_f.write_text(NODES_XML,       encoding="utf-8")
    edges_f.write_text(EDGES_XML,       encoding="utf-8")
    conns_f.write_text(CONNECTIONS_XML, encoding="utf-8")

    cmd = [
        str(netconvert),
        "--node-files",       str(nodes_f),
        "--edge-files",       str(edges_f),
        "--connection-files", str(conns_f),
        "--output-file",      str(out_f),
        "--no-turnarounds",
        "--offset.disable-normalization",
        "--junctions.corner-detail", "5",
        "--tls.default-type", "static",
        "--verbose",
    ]
    print("Running netconvert...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:\n", result.stderr[-3000:])
        sys.exit(1)
    # Print last few lines of netconvert output
    lines = (result.stderr or result.stdout or "").strip().splitlines()
    for line in lines[-5:]:
        print(line)

    patch_tls(out_f)

    for f in [nodes_f, edges_f, conns_f]:
        f.unlink(missing_ok=True)

    print(f"\nDone: {out_f.name} rebuilt.")


if __name__ == "__main__":
    main()
