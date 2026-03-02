"""Parse tripinfo.xml and calculate Average Travel Time (ATT) metrics."""

import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class TripMetrics:
    avg_travel_time: float
    total_trips: int
    departed: int
    arrived: int
    total_duration: float
    
    def __str__(self) -> str:
        return (
            f"📊 Trip Metrics:\n"
            f"  Total trips: {self.total_trips}\n"
            f"  Departed: {self.departed}\n"
            f"  Arrived: {self.arrived}\n"
            f"  Total duration sum: {self.total_duration:.2f}s\n"
            f"  ⏱️  Average Travel Time: {self.avg_travel_time:.2f}s"
        )


def parse_tripinfo(xml_path: Path | str) -> Optional[TripMetrics]:
    """Parse SUMO tripinfo.xml and calculate metrics."""
    xml_path = Path(xml_path)
    
    if not xml_path.exists():
        print(f"❌ File not found: {xml_path}")
        return None
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"❌ Error parsing XML: {e}")
        return None
    
    total_duration = 0.0
    trip_count = 0
    departed_count = 0
    arrived_count = 0
    
    for tripinfo in root.findall("tripinfo"):
        try:
            duration = float(tripinfo.get("duration", 0))
            departed = tripinfo.get("depart", None)
            arrival = tripinfo.get("arrival", None)
            
            if duration > 0:
                total_duration += duration
                trip_count += 1
                
            if departed is not None:
                departed_count += 1
            if arrival is not None and arrival != "-1":
                arrived_count += 1
                
        except (ValueError, AttributeError) as e:
            print(f"⚠️  Skipping malformed tripinfo: {e}")
            continue
    
    if trip_count == 0:
        print("⚠️  No valid trips found in XML")
        return None
    
    avg_travel_time = total_duration / trip_count if trip_count > 0 else 0
    
    return TripMetrics(
        avg_travel_time=avg_travel_time,
        total_trips=trip_count,
        departed=departed_count,
        arrived=arrived_count,
        total_duration=total_duration,
    )


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_metrics.py <tripinfo.xml>")
        print("\nExample:")
        print("  python evaluate_metrics.py scenarios/grid_3x3_lanes3_dynamic_dense/tripinfos.xml")
        sys.exit(1)
    
    xml_file = Path(sys.argv[1])
    metrics = parse_tripinfo(xml_file)
    
    if metrics:
        print(metrics)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
