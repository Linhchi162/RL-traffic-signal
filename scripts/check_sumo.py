import os
import shutil
import sys


def main() -> int:
    print("Python:", sys.version.replace("\n", " "))
    print("SUMO_HOME:", os.environ.get("SUMO_HOME", "<not set>"))

    # Check that the SUMO binaries are discoverable either via PATH or SUMO_HOME.
    for exe in ("sumo", "sumo-gui"):
        resolved = shutil.which(exe)
        print(f"{exe}:", resolved or "<not found on PATH>")

    try:
        import traci  # noqa: F401
        import sumolib  # noqa: F401

        print("TraCI import:", "OK")
        print("sumolib import:", "OK")
    except Exception as exc:  # pragma: no cover
        print("Import error:", repr(exc))
        return 2

    if shutil.which("sumo") is None and shutil.which("sumo-gui") is None and not os.environ.get("SUMO_HOME"):
        print(
            "\nSUMO binaries not found. Install SUMO and either:\n"
            "- add SUMO 'bin' to PATH, or\n"
            "- set SUMO_HOME to the SUMO install directory (containing 'bin' and 'tools')."
        )
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
