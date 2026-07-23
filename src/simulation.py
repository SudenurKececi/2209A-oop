# -*- coding: utf-8 -*-
"""
Simülasyon Çekirdeği
====================
Tek bir görev (episode) koşumu: EKF-SLAM + haritalama + Dynamic A* (global)
+ DWA (lokal) + isteğe bağlı PPO uyarlanabilir ağırlıklar.

Yöntemler:
  - "hybrid"  : Dynamic A* + DWA + PPO uyarlanabilir ağırlıklar (önerilen)
  - "classic" : Dynamic A* + DWA sabit ağırlıklar (geleneksel)
  - "apf"     : Yapay Potansiyel Alan (geleneksel taban çizgisi)
"""

import numpy as np

from environment import OfficeEnvironment, Robot, Human
from ekf_slam import EKFSLAM, OccupancyMapper
from planners import DynamicAStar, APFPlanner
from dwa import DWAPlanner
from ppo import PPOAgent, build_state


class EpisodeResult:
    def __init__(self):
        self.success = False
        self.collision = False
        self.collision_type = ""
        self.timeout = False
        self.time = 0.0
        self.path_length = 0.0
        self.energy = 0.0
        self.min_clearance = np.inf
        self.replans = 0
        self.loc_error_mean = 0.0
        self.trajectory = []

    def as_dict(self):
        return dict(success=int(self.success), collision=int(self.collision),
                    timeout=int(self.timeout), time=round(self.time, 2),
                    path_length=round(self.path_length, 2),
                    energy=round(self.energy, 3),
                    min_clearance=round(float(self.min_clearance), 3),
                    replans=self.replans,
                    loc_error_mean=round(self.loc_error_mean, 4))


def run_episode(scenario=0, method="hybrid", agent=None, seed=None,
                max_time=90.0, dt=0.1, record=False, ppo_buffer=None,
                deterministic=True, on_step=None, hold_on_goal=False,
                charge_station=None, start_battery=None):
    """Tek görev koşumu. ppo_buffer verilirse PPO eğitim verisi toplanır."""
    rng = np.random.default_rng(seed)
    env = OfficeEnvironment(scenario=scenario, seed=seed)
    robot = Robot(env.start[0], env.start[1],
                  theta=np.arctan2(env.goal[1] - env.start[1], env.goal[0] - env.start[0]))

    ekf = EKFSLAM(robot.x, robot.y, robot.theta)
    mapper = OccupancyMapper(env.width, env.height, env.res)
    astar = DynamicAStar(env.res, robot_radius=Robot.RADIUS)
    dwa = DWAPlanner(dt=dt)
    apf = APFPlanner()

    if start_battery is not None:
        robot.battery = float(start_battery)
    charge_state = "normal"     # normal | to_station | charging
    saved_goal = None

    res = EpisodeResult()
    loc_errors = []
    battery0 = robot.battery
    goal = env.goal
    n_beams, max_range = 48, 5.0

    # İlk tarama + ilk global plan (bilinen statik harita ile başlanmaz;
    # keşfedilen harita + statik grid karışımı: gerçekte robotlara kat planı verilir)
    known_grid = env.static_grid.copy()   # kat planı önceden bilinir (ofis planı)
    global_path = None
    if method in ("hybrid", "classic"):
        global_path = astar.plan(known_grid, (robot.x, robot.y), goal)

    step_count = 0
    last_replan_t = -10.0
    last_progress_t = 0.0
    pos_history = []
    ppo_state = None
    ppo_action = None
    ppo_every = 5  # PPO her 5 adımda bir ağırlık günceller
    prev_goal_dist = np.hypot(goal[0] - robot.x, goal[1] - robot.y)

    t = 0.0
    while t < max_time:
        # ---------------- Algılama ----------------
        ranges, angles = env.lidar_scan(robot.x, robot.y, robot.theta,
                                        n_beams=n_beams, max_range=max_range)
        est = ekf.pose
        # DWA için engel noktaları sensör çerçevesinde (robot merkezli) tutarlıdır:
        # ışın açıları robotun gerçek yönelimiyle ölçülür, DWA da aynı çerçevede planlar.
        valid = ranges < max_range - 0.15
        obstacle_pts = np.stack([robot.x + ranges * np.cos(angles),
                                 robot.y + ranges * np.sin(angles)], axis=1)[valid]

        # Haritalama (log-odds) — keşfedilen engeller bilinen gride eklenir
        if step_count % 3 == 0 and method in ("hybrid", "classic"):
            mapper.integrate_scan(est, ranges, angles - robot.theta + est[2], max_range)
            # Yüksek eşik: yalnızca kalıcı (statik) engeller global haritaya girer;
            # geçici engeller (insanlar) DWA'nın lokal kaçınmasına bırakılır.
            discovered = mapper.occupancy_grid(threshold=4.0)
            known_grid = np.maximum(env.static_grid, discovered)

        # ---------------- Global yeniden planlama (Dynamic A*) ----------------
        # 1) Kalıcı yeni engel yolu kapattıysa yeniden planla (statik değişim).
        # 2) Robot takıldıysa (sürekli ilerleme yoksa) keşfedilen haritayla
        #    alternatif rota planla (dinamik tıkanıklıktan kaçış).
        if method in ("hybrid", "classic") and step_count % 10 == 0:
            replan_needed = astar.needs_replan(env.static_grid, (est[0], est[1]))
            stuck = (t - last_progress_t) > 4.0
            if (replan_needed or stuck) and (t - last_replan_t) >= 4.0:
                grid_for_plan = known_grid if stuck else env.static_grid
                p = astar.replan(grid_for_plan, (est[0], est[1]), goal)
                if p is not None:
                    global_path = p
                    last_replan_t = t
                    last_progress_t = t

        # ---------------- PPO uyarlanabilir ağırlıklar ----------------
        if method == "hybrid" and agent is not None and step_count % ppo_every == 0:
            state = build_state(robot, goal, ranges, env.humans, global_path)
            if ppo_buffer is not None and ppo_state is not None:
                # Önceki karar için ödül: hedefe ilerleme - tehlike cezası
                d_now = np.hypot(goal[0] - robot.x, goal[1] - robot.y)
                progress = prev_goal_dist - d_now
                danger = max(0.0, 0.5 - np.min(ranges)) * 2.0
                reward = 2.0 * progress - danger - 0.02
                ppo_buffer['states'].append(ppo_state)
                ppo_buffer['actions'].append(ppo_action)
                ppo_buffer['logps'].append(ppo_logp)
                ppo_buffer['values'].append(agent.value(ppo_state))
                ppo_buffer['rewards'].append(reward)
                ppo_buffer['dones'].append(0.0)
                prev_goal_dist = d_now
            a, logp = agent.act(state, deterministic=deterministic)
            wts = agent.weights_from_action(a)
            dwa.set_weights(*wts)
            ppo_state, ppo_action, ppo_logp = state, a, logp

        # ---------------- Şarj istasyonu yönetimi ----------------
        if charge_station is not None:
            d_stat = np.hypot(charge_station[0] - robot.x, charge_station[1] - robot.y)
            if charge_state == "normal" and robot.battery < 30.0:
                # Batarya kritik: mevcut görevi kaydet, istasyona yönel
                saved_goal = goal.copy()
                goal = np.array(charge_station, dtype=float)
                env.goal = goal.copy()
                if method in ("hybrid", "classic"):
                    p = astar.plan(known_grid, (est[0], est[1]), goal)
                    if p is not None:
                        global_path = p
                charge_state = "to_station"
            elif charge_state == "to_station" and d_stat < 0.55:
                charge_state = "charging"
            elif charge_state == "charging":
                robot.battery = min(100.0, robot.battery + 14.0 * dt)
                if robot.battery >= 95.0:
                    charge_state = "normal"
                    if saved_goal is not None:
                        goal = saved_goal
                        env.goal = goal.copy()
                        if method in ("hybrid", "classic"):
                            p = astar.plan(known_grid, (est[0], est[1]), goal)
                            if p is not None:
                                global_path = p
                        saved_goal = None

        # ---------------- Kontrol ----------------
        human_states = [(h.pos.copy(), h.vel.copy()) for h in env.humans]
        arrived = np.hypot(goal[0] - robot.x, goal[1] - robot.y) < 0.45
        if charge_state == "charging":
            v_cmd, w_cmd = 0.0, 0.0     # istasyonda şarj oluyor
        elif hold_on_goal and arrived:
            # İnteraktif mod: hedefe varınca dur, yeni hedef bekle
            v_cmd, w_cmd = 0.0, 0.0
        elif method == "apf":
            v_cmd, w_cmd = apf.command(robot, goal, ranges, angles)
        else:
            # DWA hedefi: global yol üzerinde ileri bakış noktası
            local_goal = _lookahead(global_path, (robot.x, robot.y), 2.0) \
                if global_path is not None else goal
            v_cmd, w_cmd, _ = dwa.command(robot, local_goal, obstacle_pts,
                                          global_path=global_path,
                                          human_states=human_states)

        # ---------------- Hareket + EKF ----------------
        v_meas, w_meas = robot.step(v_cmd, w_cmd, dt, rng=rng)
        ekf.predict(v_meas, w_meas, dt)
        if step_count % 2 == 0:
            obs = env.observe_landmarks(robot.x, robot.y, robot.theta)
            ekf.update(obs)
        loc_errors.append(ekf.localization_error(robot.pose))

        env.step(dt, robot_pos=(robot.x, robot.y))
        t += dt
        step_count += 1

        # ---------------- Canlı görüntüleme / interaktif geri çağrı ----------------
        if on_step is not None:
            cmd = on_step(env=env, robot=robot, t=t, ranges=ranges, angles=angles,
                          global_path=global_path, est=ekf.pose,
                          mapper=mapper, ekf=ekf, charge_state=charge_state)
            if isinstance(cmd, dict):
                if "goal" in cmd:
                    if charge_state != "normal":
                        # Şarj sürecindeyse yeni görev şarj sonrasına ertelenir
                        saved_goal = np.array(cmd["goal"], dtype=float)
                    else:
                        # Tıklama ile yeni hedef: güncelle, rotayı yeniden planla
                        goal = np.array(cmd["goal"], dtype=float)
                        env.goal = goal.copy()
                        if method in ("hybrid", "classic"):
                            p = astar.plan(known_grid, (est[0], est[1]), goal)
                            if p is not None:
                                global_path = p
                        last_replan_t = t
                        last_progress_t = t
                        pos_history.clear()
                if "add_human" in cmd:
                    hx, hy = cmd["add_human"]
                    env.humans.append(_make_human(env, hx, hy))
                if "remove_human" in cmd and env.humans:
                    hx, hy = cmd["remove_human"]
                    dists = [np.hypot(h.pos[0] - hx, h.pos[1] - hy) for h in env.humans]
                    env.humans.pop(int(np.argmin(dists)))
                if "set_battery" in cmd:
                    robot.battery = float(cmd["set_battery"])
                if "obstacle" in cmd:
                    # Sağ tık ile yeni engel: ortama kutu ekle (Dynamic A* testi)
                    ox, oy = cmd["obstacle"]
                    rect = (ox - 0.4, oy - 0.4, 0.8, 0.8)
                    env.furniture.append(rect)
                    env.static_grid = env._build_static_grid()
                    new_lms = np.array([(rect[0], rect[1]),
                                        (rect[0] + rect[2], rect[1]),
                                        (rect[0], rect[1] + rect[3]),
                                        (rect[0] + rect[2], rect[1] + rect[3])])
                    env.landmarks = np.vstack([env.landmarks, new_lms])

        # ---------------- Metrikler / bitiş koşulları ----------------
        if record:
            res.trajectory.append((robot.x, robot.y, robot.theta, t,
                                   [(h.pos[0], h.pos[1]) for h in env.humans]))
        clearance = float(np.min(ranges))
        res.min_clearance = min(res.min_clearance, clearance)

        # İlerleme takibi (takılma tespiti): son 4 s içinde yer değiştirme
        pos_history.append((t, robot.x, robot.y))
        while pos_history and pos_history[0][0] < t - 4.0:
            pos_history.pop(0)
        if pos_history:
            t0, x0, y0 = pos_history[0]
            if np.hypot(robot.x - x0, robot.y - y0) > 0.5 or (t - t0) < 3.9:
                last_progress_t = t

        if env.collision(robot.x, robot.y, Robot.RADIUS):
            res.collision = True
            res.collision_type = "static" if env.static_collision(robot.x, robot.y, Robot.RADIUS) else "human"
            break
        d_goal = np.hypot(goal[0] - robot.x, goal[1] - robot.y)
        if d_goal < 0.4:
            res.success = True
            if not hold_on_goal:   # interaktif modda durup yeni hedef beklenir
                break
    else:
        res.timeout = True

    # PPO terminal ödülü
    if ppo_buffer is not None and ppo_state is not None:
        terminal = 10.0 if res.success else (-10.0 if res.collision else -3.0)
        ppo_buffer['states'].append(ppo_state)
        ppo_buffer['actions'].append(ppo_action)
        ppo_buffer['logps'].append(ppo_logp)
        ppo_buffer['values'].append(agent.value(ppo_state))
        ppo_buffer['rewards'].append(terminal)
        ppo_buffer['dones'].append(1.0)

    res.time = t
    res.path_length = robot.distance_traveled
    res.energy = battery0 - robot.battery
    res.replans = astar.replan_count
    res.loc_error_mean = float(np.mean(loc_errors)) if loc_errors else 0.0
    return res, env


def _make_human(env, x, y):
    """Tıklanan noktada, çevresinde serbest devriye rotası olan insan üret."""
    x = float(np.clip(x, 0.5, env.width - 0.5))
    y = float(np.clip(y, 0.5, env.height - 0.5))
    wps = [(x, y)]
    tries = 0
    while len(wps) < 4 and tries < 60:
        tries += 1
        px = float(np.clip(x + env.rng.uniform(-2.5, 2.5), 0.5, env.width - 0.5))
        py = float(np.clip(y + env.rng.uniform(-2.5, 2.5), 0.5, env.height - 0.5))
        mx, my = (wps[-1][0] + px) / 2, (wps[-1][1] + py) / 2
        if not env.static_collision(px, py, 0.35) and \
           not env.static_collision(mx, my, 0.30):
            wps.append((px, py))
    if len(wps) < 2:                      # çevre çok dar: yerinde bekleyen insan
        wps.append((x + 0.1, y + 0.1))
    return Human(wps, speed=float(env.rng.uniform(0.45, 0.75)), rng=env.rng)


def _lookahead(path, pos, dist):
    """Global yol üzerinde robottan ~dist ileride hedef nokta."""
    if path is None or len(path) == 0:
        return None
    p = np.asarray(pos)
    d = np.linalg.norm(path - p, axis=1)
    i0 = int(np.argmin(d))
    acc = 0.0
    for i in range(i0, len(path) - 1):
        acc += np.linalg.norm(path[i + 1] - path[i])
        if acc >= dist:
            return path[i + 1]
    return path[-1]
