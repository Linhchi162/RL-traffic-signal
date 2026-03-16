"""
encoder_obs.py — StateCompressor

Giảm chiều không gian trạng thái từ 19D xuống latent_dim D
bằng Autoencoder được huấn luyện tự động trong quá trình chạy.

Quy trình:
    1. Lần đầu khởi tạo → thu thập 10000 mẫu bằng cách chạy mô phỏng
       ngắn với IntersectionStateExtractor (19D).
    2. Huấn luyện Autoencoder trên các mẫu đó.
    3. Lưu encoder đã train; các lần chạy sau tự động nạp lại.
    4. Tất cả agent cùng latent_dim chia sẻ một encoder duy nhất.

Các biến thể kích thước latent:
    - CompressedState_4D   (latent_dim=4)
    - CompressedState_8D   (latent_dim=8)
    - CompressedState_16D  (latent_dim=16)  ← khuyến nghị
    - CompressedState_19D  (không nén)
    - CompressedState_32D  (latent_dim=32)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from torch.utils.data import DataLoader, TensorDataset

from .state_builder import IntersectionStateExtractor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kiến trúc mạng neural
# ---------------------------------------------------------------------------

class CompressorNet(nn.Module):
    """Encoder: nén vector 19D → latent_dim D."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StateDecoder(nn.Module):
    """Decoder: tái tạo lại vector 19D từ latent_dim D."""

    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CompressionModel(nn.Module):
    """Autoencoder hoàn chỉnh: encoder + decoder."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = CompressorNet(input_dim, latent_dim)
        self.decoder = StateDecoder(latent_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# StateCompressor — observation class
# ---------------------------------------------------------------------------

class StateCompressor:
    """
    Observation function dùng Autoencoder để nén trạng thái 19D → latent_dim D.

    Tương thích với TrafficControlEnv như một obs_class.
    Tất cả instance cùng latent_dim dùng chung một trained encoder.
    """

    _lock = threading.Lock()
    _trained_encoders: dict = {}   # latent_dim → CompressorNet
    _training_done: dict = {}      # latent_dim → bool

    def __init__(
        self,
        controller,
        latent_dim: int = 16,
        buffer_size: int = 10_000,
        num_epochs: int = 30,
        save_dir: str = "",
    ):
        """
        Args:
            controller: Đối tượng SignalController.
            latent_dim: Số chiều không gian nén.
            buffer_size: Số mẫu thu thập để huấn luyện AE.
            num_epochs: Số epoch huấn luyện AE.
            save_dir: Thư mục lưu/tải encoder đã train.
        """
        self.ctrl = controller
        self.env = controller.env
        self.latent_dim = latent_dim
        self.buffer_size = buffer_size
        self.num_epochs = num_epochs
        self.input_dim = 19
        self.batch_size = 256
        self.lr = 1e-3

        # Thiết lập đường dẫn lưu trữ
        if not save_dir:
            save_dir = os.path.join(
                os.path.dirname(__file__), "..", "ae_checkpoints"
            )
        os.makedirs(save_dir, exist_ok=True)
        self.model_path = os.path.join(save_dir, f"encoder_{latent_dim}d.pth")
        self.log_path = os.path.join(save_dir, "ae_reconstruction_log.csv")

        # Observation function gốc (19D) làm raw input
        self._raw_obs = IntersectionStateExtractor(controller)

        # Thử tải model đã huấn luyện
        self._try_load_encoder()

    # ------------------------------------------------------------------
    # Tải / Lưu encoder
    # ------------------------------------------------------------------

    def _try_load_encoder(self):
        """Tải encoder từ file nếu có; bỏ qua nếu không tồn tại."""
        with StateCompressor._lock:
            if self.latent_dim in StateCompressor._training_done:
                return
            if os.path.exists(self.model_path):
                try:
                    enc = CompressorNet(self.input_dim, self.latent_dim)
                    enc.load_state_dict(
                        torch.load(self.model_path, map_location="cpu")
                    )
                    enc.eval()
                    StateCompressor._trained_encoders[self.latent_dim] = enc
                    StateCompressor._training_done[self.latent_dim] = True
                    logger.info("Nạp encoder %dD từ '%s'.", self.latent_dim, self.model_path)
                except Exception as exc:
                    logger.warning("Không nạp được encoder: %s. Sẽ train lại.", exc)

    # ------------------------------------------------------------------
    # Huấn luyện Autoencoder
    # ------------------------------------------------------------------

    def _run_autoencoder_training(self):
        """Thu thập dữ liệu và huấn luyện Autoencoder."""
        with StateCompressor._lock:
            if self.latent_dim in StateCompressor._training_done:
                return

            logger.info("=== Bắt đầu huấn luyện Autoencoder %dD ===", self.latent_dim)

            # --- Thu thập mẫu ---
            original_cls = self.env.obs_class
            try:
                self.env.obs_class = IntersectionStateExtractor
                self.env.reset()
                raw_samples = list(self.env._gather_states().values())

                steps_needed = self.buffer_size // max(len(self.env.signal_ids), 1)
                for _ in range(steps_needed):
                    if getattr(self.env, "single_agent", True):
                        a = self.env.action_space.sample()
                        step_result = self.env.step(a)
                        obs = step_result[0]
                        raw_samples.append(obs)
                    else:
                        a = {sid: self.env.action_space.sample() for sid in getattr(self.env, "signal_ids", [])}
                        step_result = self.env.step(a)
                        obs_dict = step_result[0]
                        if isinstance(obs_dict, dict):
                            raw_samples.extend(obs_dict.values())
                        else:
                            raw_samples.append(obs_dict)

                logger.info("Thu thập %d mẫu huấn luyện.", len(raw_samples))
            finally:
                self.env.obs_class = original_cls

            # --- Huấn luyện AE ---
            t_start = time.time()
            data = torch.from_numpy(np.array(raw_samples, dtype=np.float32))
            loader = DataLoader(
                TensorDataset(data), batch_size=self.batch_size, shuffle=True
            )
            model = CompressionModel(self.input_dim, self.latent_dim)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.lr)

            model.train()
            for epoch in range(self.num_epochs):
                for (batch,) in loader:
                    optimizer.zero_grad()
                    out = model(batch)
                    loss = criterion(out, batch)
                    loss.backward()
                    optimizer.step()

            # --- Đánh giá MSE ---
            model.eval()
            total_mse = 0.0
            with torch.no_grad():
                for (batch,) in loader:
                    total_mse += criterion(model(batch), batch).item()
            avg_mse = total_mse / len(loader)
            logger.info("Reconstruction MSE: %.8f | Thời gian: %.1fs",
                        avg_mse, time.time() - t_start)
            self._log_result(avg_mse)

            # --- Lưu encoder ---
            model.encoder.eval()
            torch.save(model.encoder.state_dict(), self.model_path)
            StateCompressor._trained_encoders[self.latent_dim] = model.encoder
            StateCompressor._training_done[self.latent_dim] = True
            logger.info("Encoder lưu tại '%s'.", self.model_path)

    def _log_result(self, mse: float):
        """Ghi MSE vào file CSV để so sánh các kích thước latent."""
        header = not os.path.exists(self.log_path)
        with open(self.log_path, "a") as f:
            if header:
                f.write("latent_dim,reconstruction_mse\n")
            f.write(f"{self.latent_dim},{mse:.8f}\n")

    # ------------------------------------------------------------------
    # Observation function interface
    # ------------------------------------------------------------------

    def __call__(self) -> np.ndarray:
        """Trả về vector latent_dim D đã nén."""
        if self.latent_dim not in StateCompressor._training_done:
            self._run_autoencoder_training()
            # Cập nhật lại obs_fn cho tất cả controller
            for ctrl in self.env.controllers.values():
                ctrl.obs_fn = self.env.obs_class(ctrl)
            return np.zeros((self.latent_dim,), dtype=np.float32)

        raw = self._raw_obs()
        tensor = torch.from_numpy(raw).float().unsqueeze(0)
        encoder = StateCompressor._trained_encoders[self.latent_dim]
        with torch.no_grad():
            compressed = encoder(tensor)
        return compressed.squeeze(0).cpu().numpy().astype(np.float32)

    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.latent_dim,), dtype=np.float32
        )


# ---------------------------------------------------------------------------
# Các lớp cụ thể theo kích thước latent
# ---------------------------------------------------------------------------

class CompressedState_4D(StateCompressor):
    def __init__(self, controller):
        super().__init__(controller, latent_dim=4)

class CompressedState_8D(StateCompressor):
    def __init__(self, controller):
        super().__init__(controller, latent_dim=8)

class CompressedState_16D(StateCompressor):
    def __init__(self, controller):
        super().__init__(controller, latent_dim=16)

class CompressedState_19D(StateCompressor):
    def __init__(self, controller):
        super().__init__(controller, latent_dim=19)

class CompressedState_32D(StateCompressor):
    def __init__(self, controller):
        super().__init__(controller, latent_dim=32)
