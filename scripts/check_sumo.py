import os
import shutil
import sys


def _resolve_sumo_binary() -> str | None:
    # Prefer PATH first because some users install SUMO globally.
    binary = shutil.which("sumo")
    if binary:
        return binary

    # Fallback to SUMO_HOME if present.
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        return None

    candidate = os.path.join(sumo_home, "bin", "sumo")
    if os.name == "nt":
        candidate += ".exe"

    return candidate if os.path.exists(candidate) else None


def main() -> int:
    print("=== SUMO environment check ===")

    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        print(f"SUMO_HOME: {sumo_home}")
    else:
        print("SUMO_HOME: not set")

    sumo_binary = _resolve_sumo_binary()
    if sumo_binary:
        print(f"SUMO binary: {sumo_binary}")
    else:
        print("SUMO binary: not found in PATH or SUMO_HOME/bin")
        print("Please install SUMO and/or set SUMO_HOME correctly.")
        return 1

    try:
        import traci  # noqa: F401
        import sumolib  # noqa: F401
    except Exception as exc:  # pragma: no cover
        print(f"TraCI/sumolib import failed: {exc}")
        print("Run: pip install -r requirements.txt")
        return 1

    print("TraCI import: OK")
    print("sumolib import: OK")
    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
