"""Multi-agent training v2 — shared-experience actor-learner and PBT.

Two modes (cfg.multi_agent.mode):

"shared" (default) — Ape-X-style actor-learner (Horgan et al. 2018):
    N lightweight actor processes step their own environments with FIXED
    per-actor exploration rates (epsilon ladder) and stream transitions to the
    main process. ONE learner owns the single replay buffer and does every
    gradient update, broadcasting fresh weights to actors periodically.
    DQN is off-policy, so learning from other actors' behavior data is sound.

"pbt" — Population Based Training (Jaderberg et al. 2017):
    N full agents train independently. Every `pbt_interval` episodes an agent
    reports its rolling mean; if it ranks in the bottom quartile of the
    population's CURRENT means, it exploits (copies weights + hyperparameters
    from a random top-quartile member) and explores (perturbs lr/gamma by
    x0.8 or x1.2). Unlike v1's best-weight broadcast, the top 75% of the
    population is never overwritten, so behavioral diversity survives.

Both modes write `history.json` into the save dir for comparison plotting.

Usage (via train.py):
    python train.py train --multi-agent --ma-mode shared --num-agents 4
    python train.py train --multi-agent --ma-mode pbt --num-agents 4
"""
from __future__ import annotations

import json
import multiprocessing as mp
import queue as queue_mod
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from config import Config, MultiAgentConfig

# fork() after torch initializes its OpenMP thread pool deadlocks children on
# their first tensor op (observed: actors parked in futex_do_wait for hours).
# spawn gives every worker a clean interpreter and torch runtime.
_mp = mp.get_context("spawn")

# Abort a run if neither a transition nor an episode result arrives for this long.
STALL_TIMEOUT_S = 300.0


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_multi_agent.py)
# ---------------------------------------------------------------------------

def epsilon_ladder(i: int, n: int, base: float = 0.4, alpha: float = 1 / 128) -> float:
    """Fixed exploration rate for actor i of n: eps_i = base * alpha**(i/(n-1)).

    Actor 0 explores hardest (eps=base) forever; actor n-1 is nearly greedy.
    """
    if n <= 1:
        return base
    return float(base * alpha ** (i / (n - 1)))


def perturb_hparams(hparams: Dict[str, float], rng: random.Random) -> Dict[str, float]:
    """PBT explore step: multiply lr and gamma by 0.8 or 1.2, clamped."""
    out = dict(hparams)
    out["lr"] = float(np.clip(hparams["lr"] * rng.choice([0.8, 1.2]), 1e-6, 1e-2))
    out["gamma"] = float(np.clip(hparams["gamma"] * rng.choice([0.8, 1.2]), 0.5, 0.997))
    return out


def pbt_partition(
    means: Dict[int, float], quantile: float
) -> Tuple[List[int], List[int]]:
    """Rank agents by current mean reward; return (bottom, top) quantile ids."""
    ranked = sorted(means, key=lambda aid: means[aid])
    k = max(1, int(len(ranked) * quantile))
    return ranked[:k], ranked[-k:]


def _drain_latest(q: mp.Queue) -> Optional[Any]:
    """Return the newest item on a queue (discarding stale ones), or None."""
    item = None
    while True:
        try:
            item = q.get_nowait()
        except queue_mod.Empty:
            return item


def _save_history(save_dir: Path, mode: str, history: List[Dict[str, Any]]) -> Path:
    path = save_dir / "history.json"
    path.write_text(json.dumps({"mode": mode, "episodes": history}, indent=1))
    return path


# ---------------------------------------------------------------------------
# Shared mode — actors
# ---------------------------------------------------------------------------

def _actor_worker(
    actor_id: int,
    cfg: Config,
    episodes: int,
    epsilon: float,
    transition_queue: mp.Queue,
    weight_queue: mp.Queue,
    result_queue: mp.Queue,
    seed: int,
) -> None:
    """Env-stepping process: epsilon-greedy acting only, no buffer, no learning."""
    torch.set_num_threads(1)  # forward-only; leave cores to the learner
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    from environment import make_env
    from neural import MarioNet

    env = make_env(cfg.env, seed=seed)
    net = MarioNet(cfg.state_dim, env.action_space.n).float()
    net.eval()

    for episode in range(episodes):
        weights = _drain_latest(weight_queue)
        if weights is not None:
            net.load_state_dict(weights)

        obs, _ = env.reset()
        total_reward, length, flag_get = 0.0, 0, False
        while True:
            if np.random.rand() < epsilon:
                action = int(np.random.randint(env.action_space.n))
            else:
                with torch.no_grad():
                    s = torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0)
                    action = int(torch.argmax(net(s, model="online"), dim=1).item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            transition_queue.put(
                (
                    np.array(obs, dtype=np.float32),
                    np.array(next_obs, dtype=np.float32),
                    action,
                    float(reward),
                    bool(done),
                )
            )
            total_reward += reward
            length += 1
            obs = next_obs
            if done or info.get("flag_get"):
                flag_get = bool(info.get("flag_get"))
                break

        result_queue.put(
            {
                "agent_id": actor_id,
                "episode": episode,
                "reward": total_reward,
                "length": length,
                "flag_get": flag_get,
                "epsilon": epsilon,
                "t": time.time(),
            }
        )

    env.close()


# ---------------------------------------------------------------------------
# Shared mode — coordinator (learner lives here, in the main process)
# ---------------------------------------------------------------------------

class SharedExperienceCoordinator:
    """N actors -> one transition queue -> one learner with the only buffer.

    Learning cadence mirrors single-agent training: one gradient step per
    `learn_every` transitions received, target sync / checkpointing handled
    by Mario.train_step(). Weights are broadcast to every actor each
    `weight_sync_steps` gradient steps.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mac: MultiAgentConfig = cfg.multi_agent

    def run(self, episodes: int, save_dir: Path) -> List[Dict[str, Any]]:
        from agent import Mario

        n = self.mac.num_agents
        per_actor = max(1, episodes // n)
        save_dir.mkdir(parents=True, exist_ok=True)

        # ~226 KB per transition (two stacked-frame arrays); bound the queue so
        # backpressure on actors caps RAM at ~230 MB instead of growing unbounded
        transition_queue: mp.Queue = _mp.Queue(maxsize=1024)
        result_queue: mp.Queue = _mp.Queue()
        weight_queues: List[mp.Queue] = [_mp.Queue() for _ in range(n)]

        mario = Mario(
            state_dim=self.cfg.state_dim,
            action_dim=len(self.cfg.env.action_space),
            save_dir=save_dir,
            config=self.cfg.agent,
            checkpoint=self.cfg.train.checkpoint,
        )

        epsilons = [
            epsilon_ladder(i, n, self.mac.eps_base, self.mac.eps_alpha)
            for i in range(n)
        ]
        processes = [
            _mp.Process(
                target=_actor_worker,
                args=(
                    i, self.cfg, per_actor, epsilons[i],
                    transition_queue, weight_queues[i], result_queue,
                    self.cfg.train.seed + i,
                ),
                daemon=True,
            )
            for i in range(n)
        ]

        initial_weights = {k: v.cpu() for k, v in mario.net.state_dict().items()}
        for q in weight_queues:
            q.put(initial_weights)
        for p in processes:
            p.start()
        print(
            f"[Shared] {n} actors x {per_actor} episodes | "
            f"eps ladder: {[round(e, 4) for e in epsilons]}"
        )

        history: List[Dict[str, Any]] = []
        transitions = 0
        grad_steps = 0
        last_broadcast = 0
        results_seen = 0
        expected = n * per_actor
        recent_loss: Optional[float] = None
        t0 = time.time()
        last_progress = time.time()

        while results_seen < expected:
            if not any(p.is_alive() for p in processes) and result_queue.empty():
                print("[Shared] All actors exited.")
                break
            if time.time() - last_progress > STALL_TIMEOUT_S:
                print(
                    f"[Shared] STALL: no transitions or results for "
                    f"{STALL_TIMEOUT_S:.0f}s — aborting run."
                )
                break

            # 1. Ingest a chunk of transitions into the single buffer
            for _ in range(256):
                try:
                    t = transition_queue.get(timeout=0.05)
                except queue_mod.Empty:
                    break
                mario.cache(*t)
                transitions += 1
                last_progress = time.time()

            # 2. One gradient step per learn_every transitions ingested
            target_steps = transitions // self.cfg.agent.learn_every
            while grad_steps < target_steps:
                q, loss = mario.train_step()
                if loss is None:
                    break  # buffer not yet at batch size
                grad_steps += 1
                if loss is not None:
                    recent_loss = loss

            # 3. Broadcast fresh weights
            if grad_steps - last_broadcast >= self.mac.weight_sync_steps:
                last_broadcast = grad_steps
                w = {k: v.cpu() for k, v in mario.net.state_dict().items()}
                for q_ in weight_queues:
                    q_.put(w)

            # 4. Collect episode results
            while True:
                try:
                    r = result_queue.get_nowait()
                except queue_mod.Empty:
                    break
                r["wall"] = r["t"] - t0
                history.append(r)
                results_seen += 1
                last_progress = time.time()
                if results_seen % 10 == 0:
                    rewards = [h["reward"] for h in history[-20:]]
                    print(
                        f"[Shared] ep {results_seen}/{expected} | "
                        f"transitions {transitions} | grad steps {grad_steps} | "
                        f"20-ep mean {np.mean(rewards):.1f} | "
                        f"loss {recent_loss if recent_loss is None else round(recent_loss, 3)}"
                    )

        # Drain queues BEFORE join: a child blocked on a full pipe never exits
        while _drain_latest(transition_queue) is not None:
            pass
        for p in processes:
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()

        mario.save()
        path = _save_history(save_dir, "shared", history)
        print(
            f"[Shared] Done: {results_seen} episodes, {transitions} transitions, "
            f"{grad_steps} grad steps in {time.time() - t0:.0f}s -> {path}"
        )
        return history


# ---------------------------------------------------------------------------
# PBT mode — workers
# ---------------------------------------------------------------------------

def _pbt_worker(
    agent_id: int,
    cfg: Config,
    save_dir: Path,
    episodes: int,
    result_queue: mp.Queue,
    command_queue: mp.Queue,
    seed: int,
) -> None:
    """Full agent (own env + buffer + learner). Reports to the coordinator every
    `pbt_interval` episodes with current weights/hyperparams; applies any
    exploit command (new weights + perturbed hyperparams) it receives."""
    torch.set_num_threads(max(1, mp.cpu_count() // cfg.multi_agent.num_agents))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    from environment import make_env
    from agent import Mario

    agent_dir = save_dir / f"agent_{agent_id}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(cfg.env, seed=seed)
    mario = Mario(
        state_dim=cfg.state_dim,
        action_dim=env.action_space.n,
        save_dir=agent_dir,
        config=cfg.agent,
    )

    interval = cfg.multi_agent.pbt_interval
    recent: List[float] = []

    for episode in range(episodes):
        cmd = _drain_latest(command_queue)
        if cmd is not None:  # exploit + explore
            mario.net.load_state_dict(cmd["weights"])
            mario.cfg.lr = cmd["hparams"]["lr"]
            mario.cfg.gamma = cmd["hparams"]["gamma"]
            for g in mario.optimizer.param_groups:
                g["lr"] = cmd["hparams"]["lr"]

        obs, _ = env.reset()
        total_reward, length, flag_get = 0.0, 0, False
        while True:
            action = mario.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            mario.cache(obs, next_obs, action, reward, done)
            mario.learn()
            total_reward += reward
            length += 1
            obs = next_obs
            if done or info.get("flag_get"):
                flag_get = bool(info.get("flag_get"))
                break

        recent.append(total_reward)
        recent = recent[-20:]
        payload: Dict[str, Any] = {
            "agent_id": agent_id,
            "episode": episode,
            "reward": total_reward,
            "length": length,
            "flag_get": flag_get,
            "mean20": float(np.mean(recent)),
            "hparams": {"lr": mario.cfg.lr, "gamma": mario.cfg.gamma},
            "t": time.time(),
        }
        # Weights ride along only at PBT events (exploit-source material)
        if episode > 0 and episode % interval == 0:
            payload["weights"] = {
                k: v.cpu() for k, v in mario.net.state_dict().items()
            }
        result_queue.put(payload)

    env.close()


# ---------------------------------------------------------------------------
# PBT mode — coordinator
# ---------------------------------------------------------------------------

class PBTCoordinator:
    """Truncation-selection PBT: at each report-with-weights, re-rank the
    population on CURRENT 20-episode means; a bottom-quartile reporter exploits
    a random top-quartile member's weights + hyperparameters, perturbed."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mac: MultiAgentConfig = cfg.multi_agent

    def run(self, episodes: int, save_dir: Path) -> List[Dict[str, Any]]:
        n = self.mac.num_agents
        per_agent = max(1, episodes // n)
        save_dir.mkdir(parents=True, exist_ok=True)

        result_queue: mp.Queue = _mp.Queue()
        command_queues: List[mp.Queue] = [_mp.Queue() for _ in range(n)]
        rng = random.Random(self.cfg.train.seed)

        processes = [
            _mp.Process(
                target=_pbt_worker,
                args=(
                    i, self.cfg, save_dir, per_agent,
                    result_queue, command_queues[i],
                    self.cfg.train.seed + i,
                ),
                daemon=True,
            )
            for i in range(n)
        ]
        for p in processes:
            p.start()
        print(f"[PBT] {n} agents x {per_agent} episodes started.")

        history: List[Dict[str, Any]] = []
        means: Dict[int, float] = {}
        latest: Dict[int, Dict[str, Any]] = {}  # last weights+hparams per agent
        results_seen = 0
        exploit_events = 0
        expected = n * per_agent
        t0 = time.time()

        last_progress = time.time()
        while results_seen < expected:
            if not any(p.is_alive() for p in processes) and result_queue.empty():
                print("[PBT] All workers exited.")
                break
            if time.time() - last_progress > STALL_TIMEOUT_S:
                print(f"[PBT] STALL: no results for {STALL_TIMEOUT_S:.0f}s — aborting run.")
                break
            try:
                r = result_queue.get(timeout=60)
            except queue_mod.Empty:
                continue
            last_progress = time.time()

            aid = r["agent_id"]
            means[aid] = r["mean20"]
            results_seen += 1

            weights = r.pop("weights", None)
            if weights is not None:
                latest[aid] = {"weights": weights, "hparams": r["hparams"]}
                bottom, top = pbt_partition(means, self.mac.pbt_quantile)
                top_available = [a for a in top if a in latest and a != aid]
                if aid in bottom and top_available:
                    src = rng.choice(top_available)
                    new_h = perturb_hparams(latest[src]["hparams"], rng)
                    command_queues[aid].put(
                        {"weights": latest[src]["weights"], "hparams": new_h}
                    )
                    exploit_events += 1
                    print(
                        f"[PBT] exploit: agent {aid} "
                        f"(mean {means[aid]:.1f}) <- agent {src} "
                        f"(mean {means[src]:.1f}) | new hparams {new_h}"
                    )

            r["wall"] = r["t"] - t0
            history.append(r)
            if results_seen % (n * 10) == 0:
                print(
                    f"[PBT] {results_seen}/{expected} | means "
                    f"{ {a: round(m, 1) for a, m in means.items()} } | "
                    f"exploits {exploit_events}"
                )

        for p in processes:
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()

        path = _save_history(save_dir, "pbt", history)
        print(f"[PBT] Done: {results_seen} episodes, {exploit_events} exploit events -> {path}")
        return history


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def run_multi_agent(cfg: Config, episodes: int, save_dir: Path) -> List[Dict[str, Any]]:
    if cfg.multi_agent.mode == "pbt":
        return PBTCoordinator(cfg).run(episodes, save_dir)
    return SharedExperienceCoordinator(cfg).run(episodes, save_dir)
