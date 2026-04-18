"""

    RETURN: (reached_target: boolean, last_distance_from_target: double, time_elapsed: double, 
    drone_travel: double, scout_travel: double, rover_travel: double, perceive_calls: int, 
    step_calls: int, stuck_events: int)

"""

import numpy as np
import matplotlib.pyplot as plt

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
    
    currentDronePosition, scout_pos, rover_pos = startPoint, startPoint, startPoint
    drone_resting = False
    
    world_map = {} 
    scout_path = [] 
    scout_breadcrumbs = [startPoint] 
    
    # CHANGE
    last_distance_from_target = distance(rover_pos, targetPoint)
    
    droneArrived = False
    
    while time_elapsed < maximumTime:
        
        if not droneArrived:
            
            targetDroneDistance = distance(currentDronePosition, targetPoint)

            if targetDroneDistance < 0.5:
                droneArrived = True
                print("Drone arrivato! Mappa generata con", len(world_map), "celle.")
                plot_all_drone_features(world_map)
            else:
                
                desiredAngleDrone = np.arctan2(targetPoint[1] - currentDronePosition[1], targetPoint[0] - currentDronePosition[0])
                
                obs = mapApi_instance.perceive("drone", currentDronePosition)
                perceive_calls += 1
                for o in obs:
                    node = (round(o.x), round(o.y))
                    if node not in world_map:
                        world_map[node] = {'features': o.features, 'safe': True}
                        
                        
                ##### LOGICA REACTIVE
                
                MAX_SLOPE = 2.0
                MIN_TEXTURE = 0.4
                
                # see what happens to the main motion cells
                desiredAngles = [desiredAngleDrone-np.deg2rad(45), desiredAngleDrone, desiredAngleDrone+np.deg2rad(45)]
                
                
                look_x = round(currentDronePosition[0] + 1.0 * np.cos(desiredAngleDrone))
                look_y = round(currentDronePosition[1] + 1.0 * np.sin(desiredAngleDrone))
                lookahead_node = (look_x, look_y)
                
                # Di base, puntiamo al target
                angleDrone = desiredAngleDrone
                
                if lookahead_node in world_map:
                    features = world_map[lookahead_node]['features']
                    slope = features.get('slope', 0.0)
                    texture = features.get('texture', 1.0)
                    
                    if slope > MAX_SLOPE or texture < MIN_TEXTURE:
                        
                        deviazione_gradi = np.random.uniform(20, 70)
                        deviazione_radianti = np.radians(deviazione_gradi)
                        
                        # Decidiamo a caso se schivare a destra o a sinistra (moltiplichiamo per 1 o -1)
                        segno = np.random.choice([-1, 1])
                        
                        angleDrone = desiredAngleDrone + (deviazione_radianti * segno)
                        
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
                    print("MERDAAA, batteria scarica in posizione:" + str(currentDronePosition))
                elif drone_resting and stepResultDrone.battery_value >= 0.51:
                    drone_resting = False
        
        # Scout part
        
        time_elapsed += dt
        
        if droneArrived:
            break
    
    return (reached_target, last_distance_from_target, time_elapsed, drone_travel, 
            scout_travel, rover_travel, perceive_calls, step_calls, stuck_events)
    

def plot_all_drone_features(world_map, grid_size=50):
    """
    Dashboard 2x2 con fix per l'overlap e griglia 50x50 dettagliata.
    Visualizza: Texture, Color, Slope e Uphill Angle.
    """
    if not world_map:
        print("La mappa è vuota!")
        return

    # 1. Preparazione dati
    x_coords, y_coords = [], []
    textures, colors, slopes, uphills = [], [], [], []

    for (x, y), data in world_map.items():
        x_coords.append(x)
        y_coords.append(y)
        f = data.get('features', {})
        textures.append(f.get('texture', 0.0))
        colors.append(f.get('color', 0.0))
        slopes.append(f.get('slope', 0.0))
        uphills.append(f.get('uphill_angle', 0.0))

    # 2. Creazione Dashboard
    # Aumentiamo leggermente l'altezza (12) per ospitare i titoli senza coprire le label
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Dashboard Sensori Drone - Analisi Griglia 50x50', fontsize=16, fontweight='bold')

    # FIX OVERLAP: hspace gestisce lo spazio verticale, wspace quello orizzontale
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Configurazioni per ogni plot
    features_data = [
        (textures, 'Texture (Aderenza)', 'Greens'),
        (colors, 'Color (Ottico)', 'plasma'),
        (slopes, 'Slope (Pendenza)', 'Reds'),
        (uphills, 'Uphill Angle (Inclinazione)', 'Blues')
    ]

    for i, (data, title, cmap_name) in enumerate(features_data):
        ax = axs.flat[i]
        
        # Disegniamo le celle
        # s=45 è una dimensione indicativa per riempire bene i quadratini in 50x50
        sc = ax.scatter(x_coords, y_coords, c=data, cmap=cmap_name, marker='s', s=45)
        ax.set_title(title)
        fig.colorbar(sc, ax=ax)

        # --- FIX GRID 50x50 ---
        # Impostiamo i limiti fissi della mappa
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)

        # Label ogni 5 o 10 unità per non affollare gli assi
        ax.set_xticks(np.arange(0, grid_size + 1, 10))
        ax.set_yticks(np.arange(0, grid_size + 1, 10))

        # Griglia fitta (ogni 1 unità): usiamo i minor ticks per disegnare i "quadratini"
        ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        
        # Disegniamo la griglia solo sui minor ticks per vedere i bordi delle celle
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        ax.set_xlabel("X (celle)")
        ax.set_ylabel("Y (celle)")

    plt.show()