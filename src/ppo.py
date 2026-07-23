# -*- coding: utf-8 -*-
"""
PPO (Proximal Policy Optimization) — Uyarlanabilir Katman
==========================================================
Derin pekiştirmeli öğrenme ajanı, ortam durumuna (engel yoğunluğu, insan
yakınlığı, hedef mesafesi, yol sapması...) bakarak DWA maliyet fonksiyonu
ağırlıklarını (alpha, beta, gamma, delta) gerçek zamanlı uyarlar.

Saf NumPy aktör-kritik MLP + clipped surrogate objective (PPO-Clip).
Böylece hibrit mimari: A* (global) + DWA (lokal) + PPO (uyarlanabilirlik).
"""

import numpy as np


class MLP:
    """2 gizli katmanlı basit MLP (tanh)."""

    def __init__(self, sizes, rng, out_scale=0.01):
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            scale = np.sqrt(2.0 / sizes[i])
            if i == len(sizes) - 2:
                scale = out_scale
            self.W.append(rng.normal(0, scale, (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))

    def forward(self, x):
        acts = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.tanh(x)
            acts.append(x)
        return x, acts

    def params(self):
        return self.W + self.b

    def grads_from(self, acts, dout):
        """Basit geri yayılım; parametre gradyanları listesi ve giriş gradyanı."""
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        d = dout
        for i in reversed(range(len(self.W))):
            a_in = acts[i]
            gW[i] = a_in.T @ d
            gb[i] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.W[i].T) * (1 - acts[i] ** 2)
        return gW + gb


class Adam:
    def __init__(self, params, lr=3e-3):
        self.lr = lr
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.t)
            vh = self.v[i] / (1 - b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + eps)


class PPOAgent:
    """PPO-Clip ajanı. Durum -> 4 boyutlu sürekli eylem (DWA ağırlıkları)."""

    STATE_DIM = 8
    ACTION_DIM = 4

    # Eylem [-1,1] -> ağırlık aralıkları
    # Klasik ayar (1.0, 1.2, 0.6, 0.8) etrafında uyarlama bandı:
    # PPO, duruma göre bu bant içinde ince ayar öğrenir.
    W_LOW = np.array([0.6, 0.8, 0.3, 0.4])
    W_HIGH = np.array([1.6, 2.0, 0.9, 1.2])

    def __init__(self, seed=0, lr=3e-3, clip_eps=0.2, gamma=0.98, lam=0.95):
        self.rng = np.random.default_rng(seed)
        self.actor = MLP([self.STATE_DIM, 32, 32, self.ACTION_DIM], self.rng)
        self.critic = MLP([self.STATE_DIM, 32, 32, 1], self.rng, out_scale=0.1)
        self.log_std = np.full(self.ACTION_DIM, -0.5)
        self.opt_a = Adam(self.actor.params() + [self.log_std], lr=lr)
        self.opt_c = Adam(self.critic.params(), lr=lr)
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam

    # ------------------------------------------------------------------
    def act(self, state, deterministic=False):
        state = np.asarray(state, dtype=float).reshape(1, -1)
        mu, _ = self.actor.forward(state)
        mu = mu[0]
        if deterministic:
            a = mu
        else:
            std = np.exp(self.log_std)
            a = mu + std * self.rng.standard_normal(self.ACTION_DIM)
        logp = self._logp(mu, a)
        return a, logp

    def weights_from_action(self, a):
        """Eylemi (tanh ile sıkıştırıp) DWA ağırlık aralığına ölçekle."""
        s = np.tanh(a)
        return self.W_LOW + (s + 1) / 2 * (self.W_HIGH - self.W_LOW)

    def value(self, state):
        v, _ = self.critic.forward(np.asarray(state).reshape(1, -1))
        return float(v[0, 0])

    def _logp(self, mu, a):
        std = np.exp(self.log_std)
        return float(-0.5 * np.sum(((a - mu) / std) ** 2 + 2 * self.log_std + np.log(2 * np.pi)))

    # ------------------------------------------------------------------
    def update(self, buf, epochs=8, batch_size=64):
        """buf: dict(states, actions, logps, rewards, dones, values)"""
        states = np.array(buf['states'])
        actions = np.array(buf['actions'])
        old_logps = np.array(buf['logps'])
        rewards = np.array(buf['rewards'])
        dones = np.array(buf['dones'], dtype=float)
        values = np.array(buf['values'] + [0.0])

        # GAE avantaj hesabı
        T = len(rewards)
        adv = np.zeros(T)
        last = 0.0
        for t in reversed(range(T)):
            nonterm = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * nonterm - values[t]
            last = delta + self.gamma * self.lam * nonterm * last
            adv[t] = last
        returns = adv + values[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idx = np.arange(T)
        for _ in range(epochs):
            self.rng.shuffle(idx)
            for s0 in range(0, T, batch_size):
                b = idx[s0:s0 + batch_size]
                self._update_batch(states[b], actions[b], old_logps[b], adv[b], returns[b])

    def _update_batch(self, S, A, old_logp, adv, ret):
        n = len(S)
        std = np.exp(self.log_std)

        # ----- Aktör -----
        mu, acts_a = self.actor.forward(S)
        diff = (A - mu) / std
        logp = -0.5 * np.sum(diff ** 2 + 2 * self.log_std + np.log(2 * np.pi), axis=1)
        ratio = np.exp(logp - old_logp)
        clipped = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        use_unclipped = ((adv >= 0) & (ratio <= 1 + self.clip_eps)) | \
                        ((adv < 0) & (ratio >= 1 - self.clip_eps))
        # dL/dlogp = -adv*ratio (unclipped aktifken), 0 aksi halde
        dlogp = np.where(use_unclipped, -adv * ratio, 0.0) / n
        # dlogp/dmu = diff/std
        dmu = dlogp[:, None] * (diff / std)
        grads_a = self.actor.grads_from(acts_a, dmu)
        # log_std gradyanı: dlogp/dlogstd = diff^2 - 1
        g_logstd = np.sum(dlogp[:, None] * (diff ** 2 - 1), axis=0)
        self.opt_a.step(self.actor.params() + [self.log_std], grads_a + [g_logstd])
        self.log_std[:] = np.clip(self.log_std, -2.0, 0.5)

        # ----- Kritik -----
        v, acts_c = self.critic.forward(S)
        dv = 2 * (v[:, 0] - ret)[:, None] / n
        grads_c = self.critic.grads_from(acts_c, dv)
        self.opt_c.step(self.critic.params(), grads_c)

    # ------------------------------------------------------------------
    def save(self, path):
        np.savez(path,
                 log_std=self.log_std,
                 **{f'aW{i}': w for i, w in enumerate(self.actor.W)},
                 **{f'ab{i}': b for i, b in enumerate(self.actor.b)},
                 **{f'cW{i}': w for i, w in enumerate(self.critic.W)},
                 **{f'cb{i}': b for i, b in enumerate(self.critic.b)})

    def load(self, path):
        d = np.load(path)
        self.log_std = d['log_std']
        for i in range(len(self.actor.W)):
            self.actor.W[i] = d[f'aW{i}']
            self.actor.b[i] = d[f'ab{i}']
        for i in range(len(self.critic.W)):
            self.critic.W[i] = d[f'cW{i}']
            self.critic.b[i] = d[f'cb{i}']


def build_state(robot, goal, ranges, humans, global_path):
    """PPO durum vektörü (8 boyut, normalize)."""
    d_goal = np.hypot(goal[0] - robot.x, goal[1] - robot.y)
    min_r = float(np.min(ranges))
    mean_r = float(np.mean(ranges))
    # En yakın insan mesafesi ve yakın insan sayısı
    if humans:
        hd = [np.hypot(h.pos[0] - robot.x, h.pos[1] - robot.y) for h in humans]
        min_h = min(hd)
        near_h = sum(1 for d in hd if d < 3.0)
    else:
        min_h, near_h = 10.0, 0
    # Global yoldan sapma
    if global_path is not None and len(global_path):
        dev = float(np.min(np.linalg.norm(global_path - np.array([robot.x, robot.y]), axis=1)))
    else:
        dev = 0.0
    return np.array([
        min(d_goal, 20.0) / 20.0,
        min_r / 5.0,
        mean_r / 5.0,
        min(min_h, 10.0) / 10.0,
        min(near_h, 5) / 5.0,
        min(dev, 3.0) / 3.0,
        robot.v / robot.MAX_V,
        abs(robot.w) / robot.MAX_W,
    ])
