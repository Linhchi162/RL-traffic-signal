"""
generate_net.py — Tạo mạng lưới SUMO 2x2 (4 nút giao)

Sử dụng:
    python sumo_nets/generate_net.py

Yêu cầu: SUMO_HOME đã được khai báo.

Output:
    sumo_nets/grid2x2.net.xml           -- mạng lưới 4 nút giao
    sumo_nets/grid2x2_train.rou.xml     -- luồng xe huấn luyện (~100k giây)
    sumo_nets/grid2x2_test.rou.xml      -- luồng xe kiểm tra (~5k giây)

Thông số lưu lượng:
    Dựa trên dữ liệu thực tế từ RLTSCQ (single-intersection calibrated):
    - W→E (hướng chính): ~436 xe/h lúc peak, ~188 xe/h lúc thấp điểm
    - E→W (ngược): ~357 xe/h lúc peak
    - N→S / S→N: ~165-260 xe/h
    - Tổng tất cả hướng tại 1 nút: ~1800 xe/h lúc peak
    Với 4 nút giao, mỗi nút nhận lượng xe tương đương.
"""

import os
import subprocess
import sys

# Fix encoding tren Windows terminal
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import xml.etree.ElementTree as ET

# Đảm bảo SUMO_HOME
SUMO_HOME = os.environ.get("SUMO_HOME", "")
if not SUMO_HOME:
    sys.exit("Lỗi: Chưa khai báo biến môi trường SUMO_HOME")

HERE = Path(__file__).parent
NET_FILE = HERE / "grid2x2.net.xml"
TRAIN_ROUTE = HERE / "grid2x2_train.rou.xml"
TEST_ROUTE  = HERE / "grid2x2_test.rou.xml"

# -------------------------------------------------------------------
# Bước 1: Tạo mạng lưới 2×2
# -------------------------------------------------------------------

def generate_network():
    """
    Gọi netgenerate để tạo lưới 2×2.

    Tham số:
        --grid.number=2     : 2×2 = 4 nút giao đèn
        --grid.length=300   : Chiều dài đoạn đường 300m (tương đương RLTSCQ)
        --lanes=2           : 2 làn mỗi hướng (tương đương RLTSCQ)
        --speed=13.89       : Tốc độ tối đa 50 km/h (50/3.6 ≈ 13.89 m/s)
        --tls.guess         : Tự đặt đèn tín hiệu tại các nút giao
        --no-turnarounds    : Không cho phép quay đầu
        --lefthand          : Giao thông bên trái (giống RLTSCQ)
    """
    netgenerate_bin = Path(SUMO_HOME) / "bin" / "netgenerate"
    # Windows: thêm .exe nếu cần
    if sys.platform == "win32":
        netgenerate_bin = netgenerate_bin.with_suffix(".exe")

    cmd = [
        str(netgenerate_bin),
        "--grid",
        "--grid.number=2",
        "--grid.length=300",
        "--grid.attach-length=200",
        "--default.lanenumber=2",
        "--default.speed=13.89",
        "--tls.guess",
        "--tls.cycle.time=90",
        "--no-turnarounds",
        "--lefthand",
        "-o", str(NET_FILE),
    ]

    print(f"Tạo mạng lưới: {NET_FILE.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        sys.exit(f"netgenerate thất bại (code {result.returncode})")
    print(f"✓ Tạo xong: {NET_FILE.name}")


# -------------------------------------------------------------------
# Bước 2: Tìm tên edges từ net.xml
# -------------------------------------------------------------------

def get_external_edges(net_file: Path) -> list:
    """
    Đọc net.xml và trả về danh sách edge bên ngoài (không phải internal).
    Dùng để xác định nguồn/đích cho random trips.
    """
    tree = ET.parse(str(net_file))
    root = tree.getroot()
    external_edges = []
    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue
        eid = edge.get("id", "")
        if eid.startswith(":"):
            continue
        external_edges.append(eid)
    return external_edges


# -------------------------------------------------------------------
# Bước 3: Tạo route file bằng randomTrips.py
# -------------------------------------------------------------------

def generate_routes(
    route_file: Path,
    duration: int,
    period: float,
    seed: int = 42,
    label: str = "",
):
    """
    Gọi randomTrips.py để sinh luồng xe.

    Args:
        route_file: Đường dẫn output .rou.xml
        duration:   Tổng thời gian mô phỏng (giây)
        period:     Khoảng cách giữa 2 xe xuất phát (giây)
                    Period = 3600 / vehsPerHour
                    Tổng ~1800 xe/h / 12 hướng ≈ 150 xe/h mỗi hướng
                    → period = 3600/150 ≈ 24s
                    Với 4 nút → có thể giảm xuống ~12s (lưu lượng cao hơn)
        seed:       Random seed
        label:      Nhãn log
    """
    random_trips = Path(SUMO_HOME) / "tools" / "randomTrips.py"
    if not random_trips.exists():
        # Thử tìm ở vị trí khác
        for candidate in [
            Path(SUMO_HOME) / "tools" / "trip" / "randomTrips.py",
            Path(SUMO_HOME) / "share" / "sumo" / "tools" / "randomTrips.py",
        ]:
            if candidate.exists():
                random_trips = candidate
                break
        else:
            sys.exit(f"Không tìm thấy randomTrips.py trong {SUMO_HOME}")

    # period=8: ~450 xe/h mỗi cặp nguồn-đích, tổng ~1800 xe/h tại mỗi nút
    # Phù hợp với dữ liệu RLTSCQ (peak ~1800 xe/h tổng)
    cmd = [
        sys.executable, str(random_trips),
        "--net-file", str(NET_FILE),
        "--route-file", str(route_file),
        "--begin", "0",
        "--end", str(duration),
        "--period", str(period),
        "--seed", str(seed),
        "--min-distance", "100",
        "--allow-fringe",
        "--validate",
    ]

    print(f"Tạo route file ({label}): {route_file.name}  [period={period}s, {int(3600/period)} veh/h/pair]")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr[:500])
        print(f"⚠ randomTrips kết thúc với code {result.returncode}. Kiểm tra file output.")
    else:
        print(f"✓ Tạo xong: {route_file.name}")


# -------------------------------------------------------------------
# Bước 4: Viết route file có cấu trúc peak/offpeak (XML thủ công)
#         nếu randomTrips không khả dụng
# -------------------------------------------------------------------

def generate_routes_manual():
    """
    Viết route XML thủ công mô phỏng profile lưu lượng RLTSCQ.

    RLTSCQ dùng 12 hướng di chuyển (3 hướng × 4 cạnh).
    Với 4 nút giao, ta dùng randomTrips cho đơn giản.
    File này chỉ là fallback nếu randomTrips không chạy được.

    Profile lưu lượng (dựa trên RLTSCQ):
        Peak (giờ cao điểm): ~450 xe/h mỗi hướng chính
        Off-peak:            ~150-250 xe/h
    """
    # Train: 3 phase (peak / normal / off-peak)
    TRAIN_DURATION = 100_000   # 100k giây ≈ 27.8 giờ
    TEST_DURATION  = 5_000

    for (route_file, total_dur) in [
        (TRAIN_ROUTE, TRAIN_DURATION),
        (TEST_ROUTE, TEST_DURATION),
    ]:
        if route_file.exists():
            continue   # đã tạo rồi
        print(f"⚠ Tạo route đơn giản (manual) cho {route_file.name}")
        # Bỏ qua — randomTrips được ưu tiên

    print("Manual route generation skipped (use randomTrips).")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    HERE.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  Tao mang luoi SUMO 2x2 (4 nut giao)")
    print("  Luu luong dua theo RLTSCQ calibrated data")
    print("=" * 55)

    # 1. Tạo mạng lưới
    if NET_FILE.exists():
        print(f"Bo qua: {NET_FILE.name} da ton tai. Xoa de tao lai.")
    else:
        generate_network()

    # 2. Tạo route file
    # Thông số lưu lượng:
    #   RLTSCQ: tổng ~1800 xe/h tại 1 nút (peak), 12 hướng
    #   → ~150 xe/h / hướng → period = 3600/150 = 24s
    #   Với 4 nút → lưu lượng giữ nguyên mỗi nút → period ≈ 24s
    #   Train: period=24 (moderate), Test: period=15 (dense/stress test)
    
    TRAIN_DURATION = 100_000   # 100k giây
    TEST_DURATION  = 5_200     # 5k giây + warmup

    if TRAIN_ROUTE.exists():
        print(f"Bo qua: {TRAIN_ROUTE.name} da ton tai.")
    else:
        generate_routes(TRAIN_ROUTE, TRAIN_DURATION, period=9.0, seed=42, label="train")
        # period=9s ~ 400 veh/h/pair

    if TEST_ROUTE.exists():
        print(f"Bo qua: {TEST_ROUTE.name} da ton tai.")
    else:
        generate_routes(TEST_ROUTE, TEST_DURATION, period=9.0, seed=123, label="test")

    print("\n" + "=" * 55)
    print("Hoan tat! Files da tao trong thu muc sumo_nets/")
    print(f"  Net  : {NET_FILE.name}")
    print(f"  Train: {TRAIN_ROUTE.name}")
    print(f"  Test : {TEST_ROUTE.name}")
    print("=" * 55)


if __name__ == "__main__":
    main()
