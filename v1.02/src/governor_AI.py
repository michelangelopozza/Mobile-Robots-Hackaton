"""
governor_1.py — Strategia di navigazione multi-robot
=====================================================
Architettura:
  • DRONE  : vola in avanscoperta lungo il corridoio start→target,
             costruendo una mappa di slope/texture/color.
             Battery management attivo: si ferma se sotto soglia.
  • SCOUT  : si muove qualche cella avanti al rover sul percorso pianificato,
             rivela stuck_event PRIMA che ci cada il rover (lui non viene
             immobilizzato) e marca le celle pericolose per il replanning.
  • ROVER  : segue il percorso A* ottimale; re-pianifica ogni volta che
             si accumulano stuck event o lo scout ha segnalato pericoli.

Funzione di costo A*:
  cost(cell) = 1 + 4·(slope/30) + 500·[stuck_cell]
  Le celle sconosciute hanno un leggero penalty di esplorazione (3.0).
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Set, Tuple

# ─────────────────────── Costanti di configurazione ──────────────────────────
DRONE_ID  = "drone_1"
SCOUT_ID  = "scout_1"
ROVER_ID  = "rover_1"

DRONE_MAX_V = 1.0    # m/s — dal RobotProfile
SCOUT_MAX_V = 0.05   # m/s
ROVER_MAX_V = 0.01   # m/s

TIME_STEP   = 0.2    # secondi per step (deve coincidere con il parametro MapAPI)
ARRIVE_THR  = 0.5    # distanza (m) per considerare un waypoint raggiunto
ARRIVE_FINAL= 0.8    # distanza (m) per considerare il target raggiunto

# Drone battery
DRONE_BAT_LOW  = 0.15   # sotto questa soglia il drone non vola
DRONE_BAT_HIGH = 0.80   # sopra questa soglia il drone può ricominciare

# A* e replanning
SLOPE_MAX_DEG   = 30.0  # stessa di MapConfig.slope_max_degrees_for_velocity
STUCK_CELL_COST = 500.0 # penalità per celle con stuck event confermato
UNKNOWN_CELL_COST = 3.0 # penalità per celle non ancora percepite
REPLAN_EVERY    = 3     # ri-pianifica ogni N stuck event sul rover

# Scout: quanti waypoint avanti rispetto al rover deve stare
SCOUT_LEAD_WPS  = 6

# Drone: perceive ogni N celle di avanzamento
DRONE_PERC_STEP = 2

# Dimensioni mappa — 50×50 da CSV (2500 celle)
MAP_W = 50
MAP_H = 50


# ═══════════════════════════════ ENTRY POINT ═════════════════════════════════
def runGovernor(api, start_point, target_point, max_time):
    """
    Parametri
    ---------
    api          : MapAPI istanziata da main.py
    start_point  : (x, y) float — posizione di partenza
    target_point : (x, y) float — obiettivo
    max_time     : float — budget temporale massimo in secondi

    Ritorna
    -------
    (success, dist, time, d_dist, s_dist, r_dist, p_calls, s_calls, stuck)
    """
    # ── Registrazione ────────────────────────────────────────────────────────
    api.register_robot(DRONE_ID, "drone")
    api.register_robot(SCOUT_ID, "scout")
    api.register_robot(ROVER_ID, "rover")

    # ── Stato missione ───────────────────────────────────────────────────────
    known_map:   Dict[Tuple[int,int], dict] = {}   # (cx,cy) → feature dict
    stuck_cells: Set[Tuple[int,int]]        = set() # celle pericolose confermate

    total_time    = 0.0
    drone_pos     = list(start_point)
    scout_pos     = list(start_point)
    rover_pos     = list(start_point)
    drone_battery = 1.0

    d_dist = s_dist = r_dist = 0.0
    p_calls = s_calls = 0
    stuck_count = 0
    stuck_since_replan = 0

    gx = int(target_point[0])
    gy = int(target_point[1])

    # ─────────────────────── Helper: perceive ────────────────────────────────
    def perceive_at(robot_id: str, pos: list) -> None:
        nonlocal p_calls
        obs = api.perceive(robot_id, tuple(pos))
        p_calls += 1
        for o in obs:
            known_map[(o.x, o.y)] = o.features

    # ─────────────────────── Helper: step ────────────────────────────────────
    def step_robot(robot_id: str, pos: list, orientation: float, max_v: float):
        """Chiama api.step, aggiorna contatori, restituisce (new_pos, StepResult)."""
        nonlocal s_calls, total_time, d_dist, s_dist, r_dist
        nonlocal stuck_count, drone_battery

        result = api.step(robot_id, tuple(pos), max_v, orientation)
        s_calls   += 1
        total_time += TIME_STEP
        av = result.actual_velocity
        new_pos = [
            pos[0] + av * math.cos(orientation) * TIME_STEP,
            pos[1] + av * math.sin(orientation) * TIME_STEP,
        ]
        if robot_id == DRONE_ID:
            d_dist       += av * TIME_STEP
            drone_battery = result.battery_value
        elif robot_id == SCOUT_ID:
            s_dist += av * TIME_STEP
            if result.is_stuck:
                # Lo scout non si immobilizza: segna la cella come pericolosa
                stuck_cells.add((int(pos[0]), int(pos[1])))
        else:  # ROVER
            r_dist += av * TIME_STEP
            if result.is_stuck:
                stuck_count += 1
                stuck_cells.add((int(pos[0]), int(pos[1])))
        return new_pos, result

    # ─────────────────────── Helper: move verso waypoint ─────────────────────
    def move_to_wp(robot_id: str, pos: list, wp, max_v: float, stop_r: float = ARRIVE_THR):
        """Un solo step verso wp. Ritorna (new_pos, arrived)."""
        dx = wp[0] - pos[0]
        dy = wp[1] - pos[1]
        if math.sqrt(dx*dx + dy*dy) < stop_r:
            return pos, True
        angle   = math.atan2(dy, dx)
        new_pos, _ = step_robot(robot_id, pos, angle, max_v)
        return new_pos, False

    # ─────────────────────── Helper: costo cella per A* ──────────────────────
    def cell_cost(cx: int, cy: int) -> float:
        if (cx, cy) in stuck_cells:
            return STUCK_CELL_COST
        feat = known_map.get((cx, cy))
        if feat is None:
            return UNKNOWN_CELL_COST
        slope_norm = min(feat.get("slope", 0.0) / SLOPE_MAX_DEG, 1.0)
        return 1.0 + 4.0 * slope_norm

    # ─────────────────────── A* su griglia intera ────────────────────────────
    def astar(sx: int, sy: int) -> List[Tuple[int,int]]:
        """Trova il percorso minimo su griglia da (sx,sy) a (gx,gy)."""
        h = lambda x, y: math.sqrt((x-gx)**2 + (y-gy)**2)
        heap = [(h(sx, sy), 0.0, sx, sy)]
        came_from: Dict[Tuple[int,int], Tuple[int,int]] = {}
        g_score: Dict[Tuple[int,int], float] = {(sx, sy): 0.0}

        while heap:
            _, cost, cx, cy = heapq.heappop(heap)
            if cx == gx and cy == gy:
                # Ricostruisce percorso
                path, node = [], (gx, gy)
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append((sx, sy))
                path.reverse()
                return path

            # Vicini 8-connessi (diagonali incluse)
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),
                           (-1,-1),(-1,1),(1,-1),(1,1)]:
                nx, ny = cx+dx, cy+dy
                if not (0 <= nx < MAP_W and 0 <= ny < MAP_H):
                    continue
                move_d = math.sqrt(dx*dx + dy*dy)
                ng     = cost + cell_cost(nx, ny) * move_d
                if ng < g_score.get((nx, ny), float('inf')):
                    g_score[(nx, ny)]  = ng
                    came_from[(nx, ny)] = (cx, cy)
                    heapq.heappush(heap, (ng + h(nx, ny), ng, nx, ny))

        # Fallback: percorso diretto (non dovrebbe mai accadere su mappa 50×50)
        return [(sx, sy), (gx, gy)]

    # ─────────────────────── Conversione cella → waypoint ────────────────────
    def to_waypoints(cells: List[Tuple[int,int]]):
        """Centro di ogni cella, poi il target esatto come ultimo punto."""
        wps = [(float(x) + 0.5, float(y) + 0.5) for x, y in cells]
        if wps:
            wps[-1] = target_point
        return wps

    # ═══════════════════ FASE 1: Drone in avanscoperta ═══════════════════════
    # Il drone percorre la diagonale start→target in tappe, rivelando il
    # terreno. Si ferma se la batteria scende sotto la soglia minima.
    perceive_at(DRONE_ID, drone_pos)

    sx0, sy0 = int(start_point[0]), int(start_point[1])
    total_diag = math.sqrt((target_point[0]-start_point[0])**2 +
                           (target_point[1]-start_point[1])**2)
    num_stops  = 8   # tappe lungo la diagonale

    last_drone_perc_cell = (int(drone_pos[0]), int(drone_pos[1]))

    for i in range(1, num_stops + 1):
        if total_time >= max_time:
            break
        if drone_battery < DRONE_BAT_LOW:
            break  # batteria esaurita: non volare oltre

        t  = i / num_stops
        wx = start_point[0] + (target_point[0] - start_point[0]) * t
        wy = start_point[1] + (target_point[1] - start_point[1]) * t

        # Vola verso la tappa
        while total_time < max_time and drone_battery >= DRONE_BAT_LOW:
            dx_ = wx - drone_pos[0]
            dy_ = wy - drone_pos[1]
            if math.sqrt(dx_*dx_ + dy_*dy_) < 0.3:
                break
            drone_pos, _ = move_to_wp(DRONE_ID, drone_pos, (wx, wy),
                                      DRONE_MAX_V, stop_r=0.3)
            cx_, cy_ = int(drone_pos[0]), int(drone_pos[1])
            # Perceive ogni DRONE_PERC_STEP celle
            if (abs(cx_ - last_drone_perc_cell[0]) +
                    abs(cy_ - last_drone_perc_cell[1])) >= DRONE_PERC_STEP:
                perceive_at(DRONE_ID, drone_pos)
                last_drone_perc_cell = (cx_, cy_)

        perceive_at(DRONE_ID, drone_pos)  # perceive a ogni tappa

    # ═══════════════════ FASE 2: A* sul mappa rivelata ═══════════════════════
    path_cells = astar(sx0, sy0)
    path_wps   = to_waypoints(path_cells)
    wp_idx     = 0   # indice waypoint corrente del rover
    scout_wp_idx = 0  # indice waypoint dello scout (sempre avanti al rover)

    # ═══════════════════ FASE 3: Navigazione ═════════════════════════════════
    perceive_at(ROVER_ID, rover_pos)
    perceive_at(SCOUT_ID, scout_pos)

    while total_time < max_time:
        # ── Check arrivo ──────────────────────────────────────────────────
        final_dist = math.sqrt((rover_pos[0]-target_point[0])**2 +
                               (rover_pos[1]-target_point[1])**2)
        if final_dist < ARRIVE_FINAL:
            break

        # ── Rover: avanza verso waypoint corrente ─────────────────────────
        if wp_idx >= len(path_wps):
            # Tutti i waypoint superati: punta direttamente al target
            rover_pos, _ = move_to_wp(ROVER_ID, rover_pos, target_point,
                                      ROVER_MAX_V, stop_r=ARRIVE_FINAL)
        else:
            twp = path_wps[wp_idx]
            dx_ = twp[0] - rover_pos[0]
            dy_ = twp[1] - rover_pos[1]
            if math.sqrt(dx_*dx_ + dy_*dy_) < ARRIVE_THR:
                wp_idx += 1
                perceive_at(ROVER_ID, rover_pos)   # perceive a ogni waypoint
            else:
                rover_pos, _ = move_to_wp(ROVER_ID, rover_pos, twp, ROVER_MAX_V)

        # ── Scout: percorri qualche waypoint avanti al rover ──────────────
        scout_wp_idx = min(wp_idx + SCOUT_LEAD_WPS, len(path_wps) - 1)
        if scout_wp_idx < len(path_wps) and total_time < max_time:
            swp = path_wps[scout_wp_idx]
            scout_pos, _ = move_to_wp(SCOUT_ID, scout_pos, swp, SCOUT_MAX_V)

        # ── Drone: scouting continuo avanti al rover (se ha batteria) ─────
        if total_time < max_time:
            if drone_battery >= DRONE_BAT_LOW:
                # Punta ~10 waypoint avanti al rover
                ahead_idx = min(wp_idx + 10, len(path_wps) - 1)
                if ahead_idx < len(path_wps):
                    awp = path_wps[ahead_idx]
                    ddx = awp[0] - drone_pos[0]
                    ddy = awp[1] - drone_pos[1]
                    if math.sqrt(ddx**2 + ddy**2) > 0.3:
                        drone_pos, _ = move_to_wp(DRONE_ID, drone_pos, awp,
                                                  DRONE_MAX_V, stop_r=0.3)
                        dcx, dcy = int(drone_pos[0]), int(drone_pos[1])
                        if (dcx, dcy) not in known_map:
                            perceive_at(DRONE_ID, drone_pos)
            else:
                # Batteria bassa: drone idle (velocity=0, ricarica lenta)
                # Non chiamare step per non sprecare tempo di missione;
                # il drone verrà riutilizzato quando la batteria risale.
                pass

        # ── Replanning: triggered da stuck event o scout warning ──────────
        need_replan = (
            stuck_count - stuck_since_replan >= REPLAN_EVERY
        )
        if need_replan:
            stuck_since_replan = stuck_count
            rx, ry   = int(rover_pos[0]), int(rover_pos[1])
            path_cells = astar(rx, ry)
            path_wps   = to_waypoints(path_cells)
            wp_idx     = 0
            scout_wp_idx = 0
            scout_pos  = list(rover_pos)   # reset scout alla posizione rover
            perceive_at(ROVER_ID, rover_pos)

    # ═══════════════════ Risultati finali ════════════════════════════════════
    final_dist = math.sqrt((rover_pos[0]-target_point[0])**2 +
                           (rover_pos[1]-target_point[1])**2)
    success    = final_dist < ARRIVE_FINAL

    return (
        success,      # target raggiunto?
        final_dist,   # distanza residua dal target
        total_time,   # tempo totale missione (s)
        d_dist,       # distanza percorsa dal drone (m)
        s_dist,       # distanza percorsa dallo scout (m)
        r_dist,       # distanza percorsa dal rover (m)
        p_calls,      # numero chiamate perceive
        s_calls,      # numero chiamate step
        stuck_count,  # eventi stuck sul rover
    )