"""
    RETURN: (reached_target: boolean, last_distance_from_target: double, time_elapsed: double, 
    drone_travel: double, scout_travel: double, rover_travel: double, perceive_calls: int, 
    step_calls: int, stuck_events: int)
"""

import numpy as np
import matplotlib.pyplot as plt
import heapq

def distance(firstPoint, secondPoint):
    xDist = secondPoint[0] - firstPoint[0]
    yDist = secondPoint[1] - firstPoint[1]
    return np.sqrt(xDist**2+yDist**2)

def runGovernor(mapApi_instance, startPoint, targetPoint, maximumTime):
    dt = 0.2
    
    reached_target = False
    time_elapsed = 0.0
    drone_travel, scout_travel, rover_travel = 0.0, 0.0, 0.0
    perceive_calls, step_calls, stuck_events = 0, 0, 0
    
    mapApi_instance.register_robot("drone", "drone")
    mapApi_instance.register_robot("scout", "scout")
    mapApi_instance.register_robot("rover", "rover")
    
    currentDronePosition, currentScoutPosition, rover_pos = startPoint, startPoint, startPoint
    drone_resting = False
    
    world_map = {}
    world_map_stuck = {} 
    scout_path = [] 
    scout_breadcrumbs = [startPoint] 
    
    global_timeline = []
    
    velocityScout = 0.05
    
    last_distance_from_target = distance(rover_pos, targetPoint)
    
    droneArrived = False
    scoutArrived = False
    roverArrived = False
    
    while time_elapsed < maximumTime:
        
        # ==========================================
        # 1. FASE DRONE
        # ==========================================
        if not droneArrived:
            
            targetDroneDistance = distance(currentDronePosition, targetPoint)

            if targetDroneDistance < 0.5:
                droneArrived = True
                print("Drone arrivato! Mappa generata con", len(world_map), "celle.")
            else:
                
                desiredAngleDrone = np.arctan2(targetPoint[1] - currentDronePosition[1], targetPoint[0] - currentDronePosition[0])
                
                obs = mapApi_instance.perceive("drone", currentDronePosition)
                perceive_calls += 1
                for o in obs:
                    node = (round(o.x), round(o.y))
                    if node not in world_map:
                        world_map[node] = {'features': o.features, 'safe': True}
                        
                best_angle = desiredAngleDrone
                min_cost = float('inf')
                angoli_da_testare = [-np.pi/3, -np.pi/6, 0.0, np.pi/6, np.pi/3]
                
                for offset in angoli_da_testare:
                    test_angle = desiredAngleDrone + offset
                    look_x = round(currentDronePosition[0] + 1.5 * np.cos(test_angle))
                    look_y = round(currentDronePosition[1] + 1.5 * np.sin(test_angle))
                    look_node = (look_x, look_y)
                    
                    costo_direzione = 0.0
                    
                    if look_node in world_map:
                        features = world_map[look_node]['features']
                        slope = features.get('slope', 0.0)
                        texture = features.get('texture', 0.0)
                        color = features.get('color', 1.0)
                        
                        if color < 0.26 or texture > 0.6 or slope > 22.0:
                            costo_direzione += 500.0
                        else:
                            costo_direzione += (texture * 5.0) + (slope * 0.1) + ((1.0 - color) * 5.0)
                        costo_direzione += abs(offset) * 2.0
                    else:
                        costo_direzione += 10.0 + (abs(offset) * 2.0)
                        
                    if costo_direzione < min_cost:
                        min_cost = costo_direzione
                        best_angle = test_angle
                
                angleDrone = best_angle
                velocityDrone = 0.0 if drone_resting else 1.0

                stepResultDrone = mapApi_instance.step("drone", currentDronePosition, velocityDrone, angleDrone)
                step_calls += 1
                
                currentDronePosition = (
                    currentDronePosition[0] + stepResultDrone.actual_velocity * np.cos(angleDrone) * dt, 
                    currentDronePosition[1] + stepResultDrone.actual_velocity * np.sin(angleDrone) * dt
                )
                drone_travel += stepResultDrone.actual_velocity * dt
                
                if stepResultDrone.battery_value < 0.10:
                    drone_resting = True
                    print(f"[{time_elapsed:.1f}s] Drone in ricarica in {currentDronePosition}")
                elif drone_resting and stepResultDrone.battery_value >= 0.51:
                    drone_resting = False
                    
        # ==========================================
        # 2. FASE SCOUT
        # ==========================================
        elif droneArrived and not scoutArrived:
            
            targetScoutDistance = distance(currentScoutPosition, targetPoint)

            if targetScoutDistance < 0.5:
                scoutArrived = True
                print("Scout arrivato al target!")
            else:
                if not scout_path:
                    print("Calcolo il percorso A* per lo Scout basato sulla mappa del Drone...")
                    scout_path = get_robust_path(currentScoutPosition, targetPoint, world_map)
                    
                    if not scout_path:
                        print("ERRORE: Nessun percorso trovato! Lo scout va dritto per tentare il salvataggio.")
                
                if scout_path:
                    nextNode = scout_path[0]
                    
                    if distance(currentScoutPosition, nextNode) < 0.2:
                        scout_path.pop(0)
                        if scout_path:
                            nextNode = scout_path[0]
                        else:
                            nextNode = targetPoint
                            
                    angleScout = np.arctan2(nextNode[1] - currentScoutPosition[1], nextNode[0] - currentScoutPosition[0])
                else:
                    angleScout = np.arctan2(targetPoint[1] - currentScoutPosition[1], targetPoint[0] - currentScoutPosition[0])
                
                stepResultScout = mapApi_instance.step("scout", currentScoutPosition, velocityScout, angleScout)
                step_calls += 1
                
                currentScoutPosition = (
                    currentScoutPosition[0] + stepResultScout.actual_velocity * np.cos(angleScout) * dt, 
                    currentScoutPosition[1] + stepResultScout.actual_velocity * np.sin(angleScout) * dt
                )
                
                currentCell = (np.round(currentScoutPosition[0]), np.round(currentScoutPosition[1]))
                
                if stepResultScout.is_stuck and currentCell not in world_map_stuck:
                    world_map_stuck[currentCell] = True
                    print("STUCK EVENT SCOUT IN " + str(currentCell))
                    stuck_events += 1
                    
                    while scout_breadcrumbs and distance(scout_breadcrumbs[-1], currentScoutPosition) < 1.0:
                        scout_breadcrumbs.pop()
    
        # ==========================================
        # 3. FASE ROVER
        # ==========================================
        elif scoutArrived and not roverArrived:
            
            targetRoverDistance = distance(rover_pos, targetPoint)
            
            if targetRoverDistance < 0.5:
                roverArrived = True
                reached_target = True
                print(f"[{time_elapsed:.1f}s] ROVER ARRIVATO AL TARGET! MISSIONE COMPIUTA!")
                break 
                
            else:
                if scout_breadcrumbs:
                    nextNode = scout_breadcrumbs[0]
                    
                    if distance(rover_pos, nextNode) < 0.2:
                        scout_breadcrumbs.pop(0)
                        if scout_breadcrumbs:
                            nextNode = scout_breadcrumbs[0]
                        else:
                            nextNode = targetPoint
                            
                    angleRover = np.arctan2(nextNode[1] - rover_pos[1], nextNode[0] - rover_pos[0])
                else:
                    angleRover = np.arctan2(targetPoint[1] - rover_pos[1], targetPoint[0] - rover_pos[0])
                
                velocityRover = 0.05 
                stepResultRover = mapApi_instance.step("rover", rover_pos, velocityRover, angleRover)
                step_calls += 1
                
                rover_pos = (
                    rover_pos[0] + stepResultRover.actual_velocity * np.cos(angleRover) * dt, 
                    rover_pos[1] + stepResultRover.actual_velocity * np.sin(angleRover) * dt
                )
                rover_travel += stepResultRover.actual_velocity * dt
                
                currentCellRover = (np.round(rover_pos[0]), np.round(rover_pos[1]))
                
                if currentCellRover in world_map_stuck:
                    print(f"[{time_elapsed:.1f}s] DISASTRO FATALE! Il Rover è finito su uno Stuck Event in {currentCellRover}.")
                    reached_target = False
                    break 
                    
        # Queste righe avvengono in tutti i tick, per tutti i robot
        global_timeline.append((currentDronePosition, currentScoutPosition, rover_pos))
        time_elapsed += dt
    
    # =======================================================
    # === FINE DEL CICLO WHILE ===
    # =======================================================
    
    print("Generazione grafico dei percorsi finali...")
    plot_final_paths(world_map, world_map_stuck, global_timeline, targetPoint)
        
    return (reached_target, last_distance_from_target, time_elapsed, drone_travel, 
            scout_travel, rover_travel, perceive_calls, step_calls, stuck_events)


def get_robust_path(start_pos, target_pos, world_map):
    start_node = (round(start_pos[0]), round(start_pos[1]))
    target_node = (round(target_pos[0]), round(target_pos[1]))

    if target_node not in world_map:
        best_node = start_node
        min_dist = float('inf')
        for node in world_map:
            dist = (node[0] - target_node[0])**2 + (node[1] - target_node[1])**2
            if dist < min_dist:
                min_dist = dist
                best_node = node
        target_node = best_node

    if start_node == target_node or start_node not in world_map:
        return []

    open_list = []
    heapq.heappush(open_list, (0, start_node))
    came_from = {start_node: None}
    g_score = {start_node: 0.0}
    directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]
    max_iterations = 10000 
    iterations = 0

    while open_list and iterations < max_iterations:
        iterations += 1
        _, current = heapq.heappop(open_list)

        if current == target_node:
            break 

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)

            if neighbor in world_map and world_map[neighbor].get('safe', True):
                features = world_map[neighbor]['features']
                slope = features.get('slope', 0.0)
                texture = features.get('texture', 0.0)
                color = features.get('color', 1.0) 
                
                step_cost = 1.0 if dx == 0 or dy == 0 else 1.414
                terrain_penalty = (texture * 5.0) + (slope * 0.1) + ((1.0 - color) * 5.0)
                
                if color < 0.35: terrain_penalty += 500.0
                if texture > 0.55: terrain_penalty += 500.0
                if slope > 20.0: terrain_penalty += 100.0
                    
                tentative_g_score = g_score[current] + step_cost + terrain_penalty

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    h_dist = ((neighbor[0] - target_node[0])**2 + (neighbor[1] - target_node[1])**2)**0.5
                    f_score = tentative_g_score + (h_dist * 1.001)
                    heapq.heappush(open_list, (f_score, neighbor))

    path = []
    curr = target_node
    while curr != start_node and curr in came_from:
        path.append(curr)
        curr = came_from[curr]
    path.reverse()
    
    return path


def plot_final_paths(world_map, world_map_stuck, global_timeline, target_point, grid_size=50):
    """
    Mostra la mappa statica finale con i percorsi effettivi eseguiti dai tre robot.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. Disegna il terreno di sfondo (Pendenza)
    x_coords, y_coords, slopes = [], [], []
    if world_map:
        for (x, y), data in world_map.items():
            x_coords.append(x)
            y_coords.append(y)
            slopes.append(data['features'].get('slope', 0.0))
        sc = ax.scatter(x_coords, y_coords, c=slopes, cmap='Reds', marker='s', s=45, alpha=0.4)
        plt.colorbar(sc, ax=ax, label='Slope (Pendenza)')

    # 2. Estrai e disegna le linee dei percorsi dalla timeline
    if global_timeline:
        # Decomponiamo la timeline in liste di coordinate X e Y per ogni robot
        dx, dy = zip(*[frame[0] for frame in global_timeline])
        sx, sy = zip(*[frame[1] for frame in global_timeline])
        rx, ry = zip(*[frame[2] for frame in global_timeline])
        
        ax.plot(dx, dy, 'b--', alpha=0.6, label='Percorso Drone')
        ax.plot(sx, sy, color='orange', linestyle='-', alpha=0.8, linewidth=2, label='Percorso Scout')
        ax.plot(rx, ry, 'g-', linewidth=4, label='Percorso Rover')
        
        # Segna il punto di partenza
        ax.plot(dx[0], dy[0], 'k^', markersize=15, label='Start')

    # 3. Segna gli ostacoli confermati dallo Scout
    if world_map_stuck:
        stuck_x = [pos[0] for pos in world_map_stuck.keys()]
        stuck_y = [pos[1] for pos in world_map_stuck.keys()]
        ax.scatter(stuck_x, stuck_y, color='black', marker='X', s=150, linewidths=2, label='Stuck Scout (Evitati)')

    # 4. Target e rifiniture
    ax.plot(target_point[0], target_point[1], 'g*', markersize=25, markeredgecolor='black', label='Target')
    
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title("Tracciato Finale della Missione Multi-Agente", fontweight="bold", fontsize=14)
    ax.set_xlabel("X (celle)")
    ax.set_ylabel("Y (celle)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()