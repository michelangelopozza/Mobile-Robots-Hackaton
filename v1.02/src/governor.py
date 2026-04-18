import math
import heapq
import matplotlib.pyplot as plt

# ==========================================
# FLAG PER LA GRAFICA (Metti False in gara!)
# ==========================================
ENABLE_PLOT = True 

def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def get_neighbors(node, world_map):
    x, y = node
    directions = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    neighbors = []
    for dx, dy in directions:
        n = (x + dx, y + dy)
        if n in world_map and world_map[n]['safe']:
            neighbors.append(n)
    return neighbors

def find_path_astar(start_pos, target_pos, world_map):
    # ... [Mantieni la stessa funzione find_path_astar di prima] ...
    start_node = (round(start_pos[0]), round(start_pos[1]))
    target_node = (round(target_pos[0]), round(target_pos[1]))

    if target_node not in world_map:
        best_node = start_node
        min_dist = float('inf')
        for node in world_map:
            if not world_map[node]['safe']: continue
            dist = heuristic(node, target_node)
            if dist < min_dist:
                min_dist = dist
                best_node = node
        target_node = best_node

    if start_node == target_node or start_node not in world_map: return []

    frontier = []
    heapq.heappush(frontier, (0, start_node))
    came_from = {start_node: None}
    cost_so_far = {start_node: 0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == target_node: break

        for next_node in get_neighbors(current, world_map):
            features = world_map[next_node]['features']
            slope = features.get('slope', 0.0)
            texture = features.get('texture', 1.0)
            cell_cost = 1.0 + (slope * 2.0) + ((1.0 - texture) * 2.0)
            
            new_cost = cost_so_far[current] + cell_cost
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(next_node, target_node)
                heapq.heappush(frontier, (priority, next_node))
                came_from[next_node] = current

    path = []
    curr = target_node
    while curr != start_node and curr in came_from:
        path.append(curr)
        curr = came_from[curr]
    path.reverse()
    return path


def runGovernor(mapApi_instance, startPoint, targetPoint, maximumTime):
    dt = 0.2
    
    reached_target = False
    time_elapsed = 0.0
    drone_travel, scout_travel, rover_travel = 0.0, 0.0, 0.0
    perceive_calls, step_calls, stuck_events = 0, 0, 0
    
    mapApi_instance.register_robot("drone", "drone")
    mapApi_instance.register_robot("scout", "scout")
    mapApi_instance.register_robot("rover", "rover")
    
    drone_pos, scout_pos, rover_pos = startPoint, startPoint, startPoint
    drone_resting = False
    
    world_map = {} 
    scout_path = [] 
    scout_breadcrumbs = [startPoint] 
    
    target_threshold = 0.5
    iteration_count = 0

    # ==========================================
    # SETUP MATPLOTLIB
    # ==========================================
    if ENABLE_PLOT:
        plt.ion() # Interactive mode on
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.canvas.manager.set_window_title('Radar Missione Hackathon')

    while time_elapsed < maximumTime:
        iteration_count += 1
        
        # --- 1. DRONE LOGIC ---
        dist_drone = heuristic(drone_pos, targetPoint)
        if dist_drone > target_threshold:
            angle_drone = math.atan2(targetPoint[1] - drone_pos[1], targetPoint[0] - drone_pos[0])
            cmd_vel_drone = 0.0 if drone_resting else 1.0
            
            obs = mapApi_instance.perceive("drone", drone_pos)
            perceive_calls += 1
            for o in obs:
                node = (round(o.x), round(o.y))
                if node not in world_map:
                    world_map[node] = {'features': o.features, 'safe': True}

            res_drone = mapApi_instance.step("drone", drone_pos, cmd_vel_drone, angle_drone)
            step_calls += 1
            
            drone_pos = (drone_pos[0] + res_drone.actual_velocity * math.cos(angle_drone) * dt,
                         drone_pos[1] + res_drone.actual_velocity * math.sin(angle_drone) * dt)
            drone_travel += res_drone.actual_velocity * dt
            
            if res_drone.battery_value < 0.10: drone_resting = True
            elif drone_resting and res_drone.battery_value >= 0.99: drone_resting = False

        # --- 2. SCOUT LOGIC ---
        dist_scout_to_target = heuristic(scout_pos, targetPoint)
        if dist_scout_to_target > target_threshold:
            if not scout_path or heuristic(scout_pos, scout_path[0]) < 0.2:
                if scout_path: scout_path.pop(0)
                if not scout_path:
                    scout_path = find_path_astar(scout_pos, targetPoint, world_map)

            if scout_path:
                next_node = scout_path[0]
                angle_scout = math.atan2(next_node[1] - scout_pos[1], next_node[0] - scout_pos[0])
                res_scout = mapApi_instance.step("scout", scout_pos, 0.05, angle_scout)
                step_calls += 1
                
                if res_scout.is_stuck:
                    stuck_node = (round(scout_pos[0] + math.cos(angle_scout)), round(scout_pos[1] + math.sin(angle_scout)))
                    if stuck_node in world_map:
                        world_map[stuck_node]['safe'] = False
                    scout_path = [] 
                else:
                    scout_pos = (scout_pos[0] + res_scout.actual_velocity * math.cos(angle_scout) * dt,
                                 scout_pos[1] + res_scout.actual_velocity * math.sin(angle_scout) * dt)
                    scout_travel += res_scout.actual_velocity * dt
                    
                    if heuristic(scout_pos, scout_breadcrumbs[-1]) > 0.5:
                        scout_breadcrumbs.append(scout_pos)

        # --- 3. ROVER LOGIC ---
        dist_rover_to_target = heuristic(rover_pos, targetPoint)
        if dist_rover_to_target < target_threshold:
            reached_target = True
            break
            
        if len(scout_breadcrumbs) > 0:
            target_crumb = scout_breadcrumbs[0]
            dist_to_crumb = heuristic(rover_pos, target_crumb)
            
            if dist_to_crumb > 0.1:
                angle_rover = math.atan2(target_crumb[1] - rover_pos[1], target_crumb[0] - rover_pos[0])
                res_rover = mapApi_instance.step("rover", rover_pos, 0.01, angle_rover)
                step_calls += 1
                
                if res_rover.is_stuck:
                    stuck_events += 1
                    
                rover_pos = (rover_pos[0] + res_rover.actual_velocity * math.cos(angle_rover) * dt,
                             rover_pos[1] + res_rover.actual_velocity * math.sin(angle_rover) * dt)
                rover_travel += res_rover.actual_velocity * dt
            else:
                if len(scout_breadcrumbs) > 1: 
                    scout_breadcrumbs.pop(0)

        time_elapsed += dt

        # ==========================================
        # AGGIORNAMENTO GRAFICA (Ogni 10 iterazioni per non laggare)
        # ==========================================
        if ENABLE_PLOT and iteration_count % 10 == 0:
            ax.clear()
            ax.set_title(f"Tempo: {time_elapsed:.1f}s | Distanza Rover: {heuristic(rover_pos, targetPoint):.1f}m")
            
            # Estrai coordinate delle celle esplorate
            safe_x = [pos[0] for pos, info in world_map.items() if info['safe']]
            safe_y = [pos[1] for pos, info in world_map.items() if info['safe']]
            unsafe_x = [pos[0] for pos, info in world_map.items() if not info['safe']]
            unsafe_y = [pos[1] for pos, info in world_map.items() if not info['safe']]

            # Disegna la mappa
            if safe_x:
                ax.scatter(safe_x, safe_y, c='lightgray', marker='s', s=15, label='Esplorato (OK)')
            if unsafe_x:
                ax.scatter(unsafe_x, unsafe_y, c='red', marker='X', s=30, label='Ostacolo (Stuck)')

            # Disegna la scia sicura
            if len(scout_breadcrumbs) > 1:
                bx, by = zip(*scout_breadcrumbs)
                ax.plot(bx, by, 'g--', alpha=0.5, label='Scia Scout')

            # Disegna i Robot
            ax.plot(*drone_pos, marker='^', color='cyan', markersize=8, label='Drone')
            ax.plot(*scout_pos, marker='o', color='orange', markersize=6, label='Scout')
            ax.plot(*rover_pos, marker='s', color='green', markersize=10, label='Rover (VIP)')
            
            # Disegna Target
            ax.plot(*targetPoint, marker='*', color='gold', markersize=15, label='Target')

            ax.legend(loc='upper left')
            plt.pause(0.001) # Pausa millimetrica necessaria per aggiornare il frame

    # --- FINE CICLO ---
    if ENABLE_PLOT:
        plt.ioff() # Spegne la modalità interattiva
        plt.close() # Chiude la finestra al termine della missione

    final_distance = heuristic(rover_pos, targetPoint)
    return (reached_target, final_distance, time_elapsed, drone_travel, 
            scout_travel, rover_travel, perceive_calls, step_calls, stuck_events)