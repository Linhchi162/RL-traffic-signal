#!/usr/bin/env python3
"""
generate_grid_network.py
Sinh SUMO network 2x2 grid (4 nut giao den tin hieu, moi huong 3 lan).

Layout (y tang len tren):
         n00       n01
          |         |
 w00 -- t00 ---+--- t01 -- e01
                |         |
 w10 -- t10 ---+--- t11 -- e11
          |         |
         s10       s11

Chay:
    python generate_grid_network.py
"""

from pathlib import Path
import subprocess
import sys

OUT_DIR = Path(__file__).parent / "nets"
PREFIX  = OUT_DIR / "grid2x2"

# ---------------------------------------------------------------------------
# Node / Edge data
# ---------------------------------------------------------------------------

JUNCTIONS = {
    "t00": (150, 450), "t01": (450, 450),
    "t10": (150, 150), "t11": (450, 150),
}
EXTERNALS = {
    "n00": (150, 600), "n01": (450, 600),
    "s10": (150,   0), "s11": (450,   0),
    "w00": (  0, 450), "w10": (  0, 150),
    "e01": (600, 450), "e11": (600, 150),
}

# (from_node, to_node)  — all edges 3 lanes, speed 13.90
EDGES = [
    # External incoming
    ("n00","t00"), ("n01","t01"),
    ("s10","t10"), ("s11","t11"),
    ("w00","t00"), ("w10","t10"),
    ("e01","t01"), ("e11","t11"),
    # External outgoing
    ("t00","n00"), ("t01","n01"),
    ("t10","s10"), ("t11","s11"),
    ("t00","w00"), ("t10","w10"),
    ("t01","e01"), ("t11","e11"),
    # Internal bidirectional pairs
    ("t00","t01"), ("t01","t00"),
    ("t10","t11"), ("t11","t10"),
    ("t00","t10"), ("t10","t00"),
    ("t01","t11"), ("t11","t01"),
]

def eid(frm, to): return f"{frm}_{to}"

# Movement map per junction:
# key = direction, value = (incoming_edge, {R:out, S:out, L:out})
# R/S/L in lefthand traffic (same convention as single intersection):
#   N approach (going S): R=W, S=S, L=E
#   S approach (going N): R=E, S=N, L=W
#   E approach (going W): R=S, S=W, L=N
#   W approach (going E): R=N, S=E, L=S

JUNCTION_APPROACHES = {
    "t00": [
        ("N", eid("n00","t00"),  {"R": eid("t00","w00"), "S": eid("t00","t10"), "L": eid("t00","t01")}),
        ("S", eid("t10","t00"),  {"R": eid("t00","t01"), "S": eid("t00","n00"), "L": eid("t00","w00")}),
        ("E", eid("t01","t00"),  {"R": eid("t00","t10"), "S": eid("t00","w00"), "L": eid("t00","n00")}),
        ("W", eid("w00","t00"),  {"R": eid("t00","n00"), "S": eid("t00","t01"), "L": eid("t00","t10")}),
    ],
    "t01": [
        ("N", eid("n01","t01"),  {"R": eid("t01","t00"), "S": eid("t01","t11"), "L": eid("t01","e01")}),
        ("S", eid("t11","t01"),  {"R": eid("t01","e01"), "S": eid("t01","n01"), "L": eid("t01","t00")}),
        ("E", eid("e01","t01"),  {"R": eid("t01","t11"), "S": eid("t01","t00"), "L": eid("t01","n01")}),
        ("W", eid("t00","t01"),  {"R": eid("t01","n01"), "S": eid("t01","e01"), "L": eid("t01","t11")}),
    ],
    "t10": [
        ("N", eid("t00","t10"),  {"R": eid("t10","w10"), "S": eid("t10","s10"), "L": eid("t10","t11")}),
        ("S", eid("s10","t10"),  {"R": eid("t10","t11"), "S": eid("t10","t00"), "L": eid("t10","w10")}),
        ("E", eid("t11","t10"),  {"R": eid("t10","s10"), "S": eid("t10","w10"), "L": eid("t10","t00")}),
        ("W", eid("w10","t10"),  {"R": eid("t10","t00"), "S": eid("t10","t11"), "L": eid("t10","s10")}),
    ],
    "t11": [
        ("N", eid("t01","t11"),  {"R": eid("t11","t10"), "S": eid("t11","s11"), "L": eid("t11","e11")}),
        ("S", eid("s11","t11"),  {"R": eid("t11","e11"), "S": eid("t11","t01"), "L": eid("t11","t10")}),
        ("E", eid("e11","t11"),  {"R": eid("t11","s11"), "S": eid("t11","t10"), "L": eid("t11","t01")}),
        ("W", eid("t10","t11"),  {"R": eid("t11","t01"), "S": eid("t11","e11"), "L": eid("t11","s11")}),
    ],
}

# linkIndex per direction (same as single intersection tll.xml):
# idx 0: S_L, 1: S_S, 2: S_R
# idx 3: E_R, 4: E_S, 5: E_L
# idx 6: N_L, 7: N_S, 8: N_R
# idx 9: W_R, 10:W_S, 11:W_L
DIR_LINK_ORDER = {
    "S": [("L",0,2,2), ("S",1,1,1), ("R",2,0,0)],
    "E": [("R",3,0,0), ("S",4,1,1), ("L",5,2,2)],
    "N": [("L",6,2,2), ("S",7,1,1), ("R",8,0,0)],
    "W": [("R",9,0,0), ("S",10,1,1), ("L",11,2,2)],
}

PHASE_STATES = [
    ("39", "GGrrrrGGrrrr"),
    ("5",  "yyrrrryyrrrr"),
    ("25", "rrGrrrrrGrrr"),
    ("5",  "rryrrrrryrrr"),
    ("31", "rrrGGrrrrGGr"),
    ("5",  "rrryyrrrryyr"),
    ("24", "rrrrrGrrrrrG"),
    ("5",  "rrrrryrrrrry"),
]

# ---------------------------------------------------------------------------
# XML generators
# ---------------------------------------------------------------------------

def write_nod():
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
             'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">']
    for nid, (x, y) in JUNCTIONS.items():
        lines.append(f'    <node id="{nid}" x="{x:.2f}" y="{y:.2f}" '
                     f'type="traffic_light" tl="{nid}"/>')
    for nid, (x, y) in EXTERNALS.items():
        lines.append(f'    <node id="{nid}" x="{x:.2f}" y="{y:.2f}" type="dead_end"/>')
    lines.append('</nodes>')
    (PREFIX.with_suffix('') if False else Path(str(PREFIX) + '.nod.xml')).write_text(
        '\n'.join(lines), encoding='utf-8')
    print("[nod] written")

def write_edg():
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
             'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">']
    for frm, to in EDGES:
        lines.append(f'    <edge id="{eid(frm,to)}" from="{frm}" to="{to}" '
                     f'priority="-1" numLanes="3" speed="13.90"/>')
    lines.append('</edges>')
    Path(str(PREFIX) + '.edg.xml').write_text('\n'.join(lines), encoding='utf-8')
    print("[edg] written")

def write_con():
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<connections xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
             'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/connections_file.xsd">']
    for jid, approaches in JUNCTION_APPROACHES.items():
        lines.append(f'    <!-- {jid} -->')
        for direction, inc_edge, moves in approaches:
            for turn, out_edge in [("R", moves["R"]), ("S", moves["S"]), ("L", moves["L"])]:
                fl = {"R": 0, "S": 1, "L": 2}[turn]
                tl = fl
                lines.append(f'    <connection from="{inc_edge}" to="{out_edge}" '
                             f'fromLane="{fl}" toLane="{tl}"/>')
    # Outgoing terminator entries
    for frm, to in EDGES:
        if frm in JUNCTIONS and to in EXTERNALS:
            lines.append(f'    <connection from="{eid(frm,to)}"/>')
    lines.append('</connections>')
    Path(str(PREFIX) + '.con.xml').write_text('\n'.join(lines), encoding='utf-8')
    print("[con] written")

def write_tll():
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<tlLogics xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
             'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/tllogic_file.xsd">']
    for jid, approaches in JUNCTION_APPROACHES.items():
        lines.append(f'    <tlLogic id="{jid}" type="static" programID="0" offset="0">')
        for dur, state in PHASE_STATES:
            lines.append(f'        <phase duration="{dur}" state="{state}"/>')
        lines.append('    </tlLogic>')
        lines.append('')

        approach_map = {d: (inc, moves) for d, inc, moves in approaches}
        for direction, (movements, link_specs) in [
            ("S", (approach_map["S"], DIR_LINK_ORDER["S"])),
            ("E", (approach_map["E"], DIR_LINK_ORDER["E"])),
            ("N", (approach_map["N"], DIR_LINK_ORDER["N"])),
            ("W", (approach_map["W"], DIR_LINK_ORDER["W"])),
        ]:
            inc_edge = movements[0]
            move_map = movements[1]
            for turn, link_idx, fl, tl in link_specs:
                out_edge = move_map[turn]
                lines.append(
                    f'    <connection from="{inc_edge}" to="{out_edge}" '
                    f'fromLane="{fl}" toLane="{tl}" tl="{jid}" linkIndex="{link_idx}"/>'
                )
        lines.append('')
    lines.append('</tlLogics>')
    Path(str(PREFIX) + '.tll.xml').write_text('\n'.join(lines), encoding='utf-8')
    print("[tll] written")


def run_netconvert():
    nod = str(PREFIX) + '.nod.xml'
    edg = str(PREFIX) + '.edg.xml'
    con = str(PREFIX) + '.con.xml'
    tll = str(PREFIX) + '.tll.xml'
    out = str(PREFIX) + '.net.xml'
    cmd = [
        "netconvert",
        f"--node-files={nod}",
        f"--edge-files={edg}",
        f"--connection-files={con}",
        f"--tllogic-files={tll}",
        f"--output-file={out}",
        "--lefthand", "true",
        "--no-turnarounds", "true",
        "--junctions.corner-detail", "0",
        "--junctions.limit-turn-speed", "-1",
        "--offset.disable-normalization", "true",
        "--geometry.min-radius.fix.railways", "false",
        "--geometry.avoid-overlap", "false",
        "--geometry.max-grade.fix", "false",
        "--rectangular-lane-cut", "false",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[netconvert] OK -> {out}")
        if result.stderr.strip():
            for line in result.stderr.strip().splitlines():
                print(f"  {line}")
    else:
        print("[netconvert] FAILED:")
        print(result.stderr)
        sys.exit(1)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_nod()
    write_edg()
    write_con()
    write_tll()
    run_netconvert()
    print("\nDone. Network: nets/grid2x2.net.xml")
