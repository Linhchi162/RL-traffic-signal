"""Minimal Graph+LSTM DQN training loop for the STMARLEnv.

This is intentionally a *starter* implementation:
- Shared policy across all TLS agents
- One graph per decision step
- Node features aggregated from incoming edges
- Directed control graph between TLS nodes

Dependencies:
- torch
- torch_geometric

PyG installation can be tricky on Windows. If import fails, follow:
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path


def _require_deps() -> None:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "Missing dependency for GNN training. Install torch + torch_geometric.\n"
            "See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html\n\n"
            f"Import error: {exc!r}"
        )


@dataclass
class Transition:
    x: "torch.Tensor"  # [N, F]
    action: "torch.Tensor"  # [N]
    reward: "torch.Tensor"  # [N]
    next_x: "torch.Tensor"  # [N, F]
    done: "torch.Tensor"  # scalar 0/1


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._data: list[Transition] = []
        self._pos = 0

    def __len__(self) -> int:
        return len(self._data)

    def push(self, tr: Transition) -> None:
        if len(self._data) < self.capacity:
            self._data.append(tr)
        else:
            self._data[self._pos] = tr
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self._data, k=min(batch_size, len(self._data)))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a GNN+LSTM DQN on the SUMO traffic environment.")
    p.add_argument("--cfg", default="scenarios/grid_3x3_lanes3_dynamic_dense/run.sumo.cfg")
    p.add_argument("--gui", action="store_true")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-decisions", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--buffer", type=int, default=5000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--train-every", type=int, default=1)
    p.add_argument("--target-every", type=int, default=200)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay", type=int, default=5000)
    p.add_argument("--hidden", type=int, default=64)
    return p.parse_args()


def main() -> int:
    _require_deps()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    from env import EnvConfig, STMARLEnv

    args = _parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    class GraphLSTMDQN(nn.Module):
        def __init__(self, in_dim: int, hidden: int, out_dim: int = 2):
            super().__init__()
            self.gcn1 = GCNConv(in_dim, hidden)
            self.gcn2 = GCNConv(hidden, hidden)
            self.lstm = nn.LSTMCell(hidden, hidden)
            self.head = nn.Linear(hidden, out_dim)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
            z = F.relu(self.gcn1(x, edge_index))
            z = F.relu(self.gcn2(z, edge_index))
            h2, c2 = self.lstm(z, (h, c))
            q = self.head(h2)
            return q, h2, c2

    # Create env (re-created each episode for clean SUMO runs)
    cfg_path = Path(args.cfg)
    edge_index_list = None
    n_nodes = None

    # Init networks with inferred dims
    in_dim = 4
    policy = GraphLSTMDQN(in_dim=in_dim, hidden=args.hidden)
    target = GraphLSTMDQN(in_dim=in_dim, hidden=args.hidden)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer)

    def epsilon(step: int) -> float:
        # Linear decay
        frac = min(1.0, step / max(1, args.eps_decay))
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    global_step = 0

    for ep in range(args.episodes):
        env = STMARLEnv(EnvConfig(sumo_cfg=cfg_path, gui=args.gui, decision_interval_s=10, yellow_s=3, seed=args.seed + ep))
        try:
            # Control graph
            if edge_index_list is None:
                edge_index_list = env.get_control_edge_index()
                edge_index = torch.tensor(edge_index_list, dtype=torch.long)
                n_nodes = len(env.control_nodes)
            else:
                edge_index = torch.tensor(edge_index_list, dtype=torch.long)

            # Recurrent state per node
            h = torch.zeros((n_nodes, args.hidden), dtype=torch.float32)
            c = torch.zeros((n_nodes, args.hidden), dtype=torch.float32)

            obs = env.observation()
            x = torch.tensor(env.node_features_from_observation(obs), dtype=torch.float32)

            ep_return = 0.0
            for k in range(args.max_decisions):
                eps = epsilon(global_step)

                with torch.no_grad():
                    q, h2, c2 = policy(x, edge_index, h, c)

                # Epsilon-greedy actions per node
                if random.random() < eps:
                    a = torch.randint(low=0, high=2, size=(n_nodes,), dtype=torch.long)
                else:
                    a = torch.argmax(q, dim=-1)

                actions = {tls: int(a[i].item()) for i, tls in enumerate(env.control_nodes)}
                next_obs, rew_map, done, _info = env.step(actions)

                r = torch.tensor([float(rew_map[tls]) for tls in env.control_nodes], dtype=torch.float32)
                next_x = torch.tensor(env.node_features_from_observation(next_obs), dtype=torch.float32)

                buffer.push(
                    Transition(
                        x=x.detach(),
                        action=a.detach(),
                        reward=r.detach(),
                        next_x=next_x.detach(),
                        done=torch.tensor(float(done), dtype=torch.float32),
                    )
                )

                ep_return += float(r.mean().item())

                # Move recurrent state forward (for action selection only)
                h, c = h2.detach(), c2.detach()
                x = next_x

                # Train
                if global_step % args.train_every == 0 and len(buffer) >= args.batch:
                    batch = buffer.sample(args.batch)
                    loss_accum = 0.0

                    for tr in batch:
                        # Fresh recurrent state for training (simple baseline; sequence replay is a later upgrade)
                        h0 = torch.zeros((n_nodes, args.hidden), dtype=torch.float32)
                        c0 = torch.zeros((n_nodes, args.hidden), dtype=torch.float32)

                        q_pred, _h, _c = policy(tr.x, edge_index, h0, c0)
                        q_a = q_pred.gather(1, tr.action.view(-1, 1)).squeeze(1)

                        with torch.no_grad():
                            q_next, _hn, _cn = target(tr.next_x, edge_index, h0, c0)
                            q_next_max = torch.max(q_next, dim=1).values
                            y = tr.reward + args.gamma * (1.0 - tr.done) * q_next_max

                        loss = F.smooth_l1_loss(q_a, y)
                        optim.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                        optim.step()
                        loss_accum += float(loss.item())

                    if global_step % 50 == 0:
                        print(f"ep={ep} step={global_step} eps={eps:.3f} loss={loss_accum/len(batch):.4f} return={ep_return:.2f}")

                if global_step % args.target_every == 0 and global_step > 0:
                    target.load_state_dict(policy.state_dict())

                global_step += 1
                if done:
                    break

            print(f"Episode {ep} finished: meanRewardSum={ep_return:.2f} buffer={len(buffer)}")

        finally:
            env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
