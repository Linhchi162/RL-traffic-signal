import argparse
import os
import subprocess
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a multi-intersection grid SUMO scenario (net + trips + routes + cfg). "
            "Requires SUMO (netgenerate) and SUMO tools (randomTrips.py)."
        )
    )
    parser.add_argument("--x", type=int, default=3, help="Grid x-number (junctions count), default: 3")
    parser.add_argument("--y", type=int, default=3, help="Grid y-number (junctions count), default: 3")
    parser.add_argument("--length", type=float, default=300.0, help="Edge length in meters, default: 300")
    parser.add_argument(
        "--lanes",
        type=int,
        default=3,
        help="Number of lanes per edge direction (default: 3).",
    )
    parser.add_argument(
        "--attach-length",
        type=float,
        default=200.0,
        help=(
            "Length (m) of streets attached to the outer grid junctions to create entry/exit roads. "
            "0 disables attached streets. Default: 200"
        ),
    )
    parser.add_argument("--end", type=float, default=3600.0, help="Simulation end time (s), default: 3600")
    parser.add_argument(
        "--period",
        type=float,
        default=2.0,
        help="Vehicle insertion period (s). Lower => more traffic. Default: 2.0",
    )
    parser.add_argument(
        "--density-schedule",
        default=None,
        help=(
            "Optional variable demand schedule for randomTrips.py using --insertion-density. "
            "Format: 'begin:end:density,begin:end:density,...' (seconds and vehicles/hour/km). "
            "Example: '0:900:4,900:1800:12,1800:2700:25,2700:3600:8'. "
            "When set, --period is ignored."
        ),
    )
    parser.add_argument(
        "--density-multiplier",
        type=float,
        default=1.0,
        help="Multiply every density value in --density-schedule by this factor (default: 1.0).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed, default: 42")
    parser.add_argument(
        "--vehicle-class",
        default="passenger",
        help=(
            "Vehicle class for generated trips (passed to randomTrips.py as --vehicle-class). "
            "Default: passenger"
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output scenario folder name under scenarios/. Default: grid_{x}x{y}",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="After generating, launch sumo-gui with the generated config.",
    )
    parser.add_argument(
        "--sumo-home",
        default=None,
        help=(
            "Optional SUMO installation directory (the folder that contains 'bin' and 'tools'). "
            "If omitted, uses SUMO_HOME env var or tries to infer it from PATH."
        ),
    )
    parser.add_argument(
        "--fringe-factor",
        default="max",
        help=(
            "Passed to randomTrips.py as --fringe-factor. Use a float (e.g. 5) to bias toward fringe edges, "
            "or 'max' to force all trips to start/end at the fringe. Default: max"
        ),
    )
    parser.add_argument(
        "--allow-fringe",
        action="store_true",
        help="Pass --allow-fringe to randomTrips.py (useful for some networks/fringe definitions).",
    )
    parser.add_argument(
        "--random-depart",
        action="store_true",
        help=(
            "Pass --random-depart to randomTrips.py (depart times randomized between begin and end). "
            "If not set, departures happen roughly every --period seconds."
        ),
    )
    parser.add_argument(
        "--random-departpos",
        action="store_true",
        help="Pass --random-departpos to randomTrips.py (randomize depart position on the edge).",
    )
    parser.add_argument(
        "--random-arrivalpos",
        action="store_true",
        help="Pass --random-arrivalpos to randomTrips.py (randomize arrival position on the edge).",
    )
    parser.add_argument(
        "--poisson",
        action="store_true",
        help="Pass --poisson to randomTrips.py (more bursty flow departures).",
    )
    return parser.parse_args()


def _infer_sumo_home(cli_sumo_home: str | None) -> Path:
    if cli_sumo_home:
        return Path(cli_sumo_home)

    env_sumo_home = os.environ.get("SUMO_HOME")
    if env_sumo_home:
        return Path(env_sumo_home)

    # Fallback: infer from PATH (common on Windows when SUMO/bin is added to PATH).
    sumo_exe = shutil.which("sumo") or shutil.which("sumo-gui")
    if sumo_exe:
        return Path(sumo_exe).resolve().parent.parent

    raise SystemExit(
        "SUMO_HOME is not set and SUMO binaries were not found on PATH.\n"
        "Fix by either:\n"
        "- setting SUMO_HOME to your SUMO install directory (contains bin/tools), or\n"
        "- adding SUMO 'bin' to PATH, or\n"
        "- passing --sumo-home <path>."
    )


def _require_random_trips(sumo_home: Path) -> Path:
    random_trips = sumo_home / "tools" / "randomTrips.py"
    if not random_trips.exists():
        raise SystemExit(f"randomTrips.py not found at: {random_trips} (SUMO_HOME={sumo_home})")
    return random_trips


def _require_binary(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise SystemExit(
            f"Required SUMO binary '{name}' not found on PATH. "
            "Add SUMO 'bin' to PATH (or install SUMO)."
        )
    return resolved


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _parse_density_schedule(schedule: str) -> list[tuple[float, float, float]]:
    segments: list[tuple[float, float, float]] = []
    for raw in schedule.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) != 3:
            raise SystemExit(
                "Invalid --density-schedule. Expected format 'begin:end:density,...' "
                "e.g. '0:900:4,900:1800:12,1800:2700:25,2700:3600:8'."
            )
        begin_s, end_s, dens_s = parts
        begin = float(begin_s)
        end = float(end_s)
        density = float(dens_s)
        if end <= begin:
            raise SystemExit(f"Invalid schedule segment (end <= begin): {raw}")
        if density <= 0:
            raise SystemExit(f"Invalid schedule segment (density must be > 0): {raw}")
        segments.append((begin, end, density))

    if not segments:
        raise SystemExit("--density-schedule provided but no segments were parsed")

    segments.sort(key=lambda x: x[0])
    return segments


def _merge_route_files(route_files: list[Path], out_file: Path) -> None:
    # randomTrips.py outputs a <routes> root. We merge children and keep only the first vType block.
    merged_root = ET.Element("routes")
    kept_vtype = False

    for idx, file_path in enumerate(route_files):
        tree = ET.parse(file_path)
        root = tree.getroot()

        for child in list(root):
            tag = child.tag
            if tag in {"vType", "vTypeDistribution"}:
                if kept_vtype:
                    continue
                kept_vtype = True
                merged_root.append(child)
                continue

            # Ensure ids are unique across segments.
            if "id" in child.attrib:
                child.attrib["id"] = f"seg{idx}_" + child.attrib["id"]
            merged_root.append(child)

    ET.ElementTree(merged_root).write(out_file, encoding="utf-8", xml_declaration=True)


def main() -> int:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    scenarios_root = repo_root / "scenarios"

    out_name = args.out or f"grid_{args.x}x{args.y}"
    scenario_dir = scenarios_root / out_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    net_file = scenario_dir / f"{out_name}.net.xml"
    trips_file = scenario_dir / f"{out_name}.trips.xml"
    routes_file = scenario_dir / f"{out_name}.rou.xml"
    cfg_file = scenario_dir / "run.sumo.cfg"

    # 1) Generate a grid network with traffic lights.
    _require_binary("netgenerate")
    netgenerate_cmd = [
        "netgenerate",
        "--grid",
        "--grid.x-number",
        str(args.x),
        "--grid.y-number",
        str(args.y),
        "--grid.length",
        str(args.length),
        "--grid.attach-length",
        str(args.attach_length),
        "--default.lanenumber",
        str(args.lanes),
        "--output-file",
        str(net_file),
        "--tls.guess",
        "--tls.default-type",
        "static",
        "--tls.cycle.time",
        "60",
        "--tls.yellow.time",
        "3",
        "--tls.allred.time",
        "0",
    ]
    _run(netgenerate_cmd, cwd=scenario_dir)

    # 2) Generate random trips and route file using SUMO tools.
    sumo_home = _infer_sumo_home(args.sumo_home)
    random_trips = _require_random_trips(sumo_home)
    if args.density_schedule:
        segments = _parse_density_schedule(args.density_schedule)
        schedule_end = max(end for _, end, _ in segments)
        if schedule_end > args.end:
            raise SystemExit(
                f"Schedule end ({schedule_end}) is greater than --end ({args.end}). "
                "Increase --end or adjust --density-schedule."
            )

        tmp_dir = scenario_dir / "_tmp_density"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        seg_route_files: list[Path] = []
        # Use a constant prefix across segments so that generated vType ids match.
        # We ensure uniqueness of vehicles/flows during merge by prefixing their ids with seg{idx}_.
        constant_prefix = "rt_"
        for i, (begin, end, density) in enumerate(segments):
            density = density * args.density_multiplier
            seg_trips = tmp_dir / f"seg{i}.trips.xml"
            seg_routes = tmp_dir / f"seg{i}.rou.xml"
            seg_route_files.append(seg_routes)

            cmd = [
                "python",
                str(random_trips),
                "-n",
                str(net_file),
                "-o",
                str(seg_trips),
                "-r",
                str(seg_routes),
                "-b",
                str(begin),
                "-e",
                str(end),
                "--insertion-density",
                str(density),
                "-s",
                str(args.seed + i),
                "--prefix",
                constant_prefix,
                "--vehicle-class",
                str(args.vehicle_class),
                "--fringe-factor",
                str(args.fringe_factor),
                "--validate",
            ]
            if args.allow_fringe:
                cmd.append("--allow-fringe")
            if args.random_depart:
                cmd.append("--random-depart")
            if args.random_departpos:
                cmd.append("--random-departpos")
            if args.random_arrivalpos:
                cmd.append("--random-arrivalpos")
            if args.poisson:
                cmd.append("--poisson")

            _run(cmd, cwd=scenario_dir)

        _merge_route_files(seg_route_files, routes_file)
    else:
        random_trips_cmd = [
            "python",
            str(random_trips),
            "-n",
            str(net_file),
            "-o",
            str(trips_file),
            "-r",
            str(routes_file),
            "-e",
            str(args.end),
            "-p",
            str(args.period),
            "-s",
            str(args.seed),
            "--prefix",
            "rt_",
            "--vehicle-class",
            str(args.vehicle_class),
            "--fringe-factor",
            str(args.fringe_factor),
            "--validate",
        ]
        if args.allow_fringe:
            random_trips_cmd.append("--allow-fringe")
        if args.random_depart:
            random_trips_cmd.append("--random-depart")
        if args.random_departpos:
            random_trips_cmd.append("--random-departpos")
        if args.random_arrivalpos:
            random_trips_cmd.append("--random-arrivalpos")
        if args.poisson:
            random_trips_cmd.append("--poisson")
        _run(random_trips_cmd, cwd=scenario_dir)

    # 3) Write a minimal SUMO config.
    cfg_file.write_text(
        """<configuration>
    <input>
        <net-file value="{net}"/>
        <route-files value="{routes}"/>
    </input>

    <output>
        <tripinfo-output value="tripinfos.xml"/>
    </output>

    <time>
        <begin value="0"/>
        <end value="{end}"/>
    </time>

    <report>
        <log value="sumo_log.txt"/>
    </report>
</configuration>
""".format(
            net=net_file.name,
            routes=routes_file.name,
            end=args.end,
        ),
        encoding="utf-8",
    )

    print(f"Generated scenario: {scenario_dir}")
    print(f"Run GUI: sumo-gui -c {cfg_file.relative_to(repo_root)}")

    if args.gui:
        _require_binary("sumo-gui")
        _run(["sumo-gui", "-c", str(cfg_file)], cwd=repo_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
