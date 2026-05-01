"""
debug_c8.py — chay truc tiep tren Vast de xem crash o buoc nao
python debug_c8.py
"""
import os, sys, faulthandler

# Bat segfault handler: neu C crash thi in traceback ra stderr
faulthandler.enable()

sys.path.insert(0, "/workspace/rl-traffic")
os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")
os.environ["LIBSUMO_AS_TRACI"] = "1"

MODEL = "./exp_grid/ddqn_pressure_s123/grid_ddqn_final_model.zip"
NET   = "./nets/cologne8/cologne8.net.xml"
ROUTE = "./nets/cologne8/cologne8.rou.xml"

print("=== debug_c8 start ===", flush=True)

print("[1] importing stable_baselines3...", flush=True)
from stable_baselines3 import DQN
print("[1] OK", flush=True)

print("[2] importing rl_controller...", flush=True)
from rl_controller.traffic_env   import TrafficControlEnv
from rl_controller.state_builder import BaselineObservation
from rl_controller.grid_env      import MultiAgentVecEnv
print("[2] OK", flush=True)

print("[3] loading model...", flush=True)
agent = DQN.load(MODEL, env=None, device="cpu")
print("[3] OK", flush=True)

print("[4] creating TrafficControlEnv (khoi dong SUMO tam thoi)...", flush=True)
def make_env():
    return TrafficControlEnv(
        net_file        = NET,
        route_file      = ROUTE,
        sim_duration    = 28800 + 99999,
        reward_fn       = "queue",
        obs_class       = BaselineObservation,
        single_agent    = False,
        show_warnings   = False,
        extra_sumo_args = "--begin 25200",
    )
env = MultiAgentVecEnv(make_env)
print("[4] OK, n_agents=%d" % env.num_envs, flush=True)

print("[5] resetting env...", flush=True)
obs = env.reset()
print("[5] OK, obs.shape=%s" % str(obs.shape), flush=True)

print("[6] chay 5 buoc...", flush=True)
for i in range(5):
    actions = agent.predict(obs, deterministic=True)[0]
    obs, rews, dones, _ = env.step(actions)
    print("  step %d rews=%s" % (i, rews), flush=True)

env.close()
print("=== debug_c8 DONE ===", flush=True)
