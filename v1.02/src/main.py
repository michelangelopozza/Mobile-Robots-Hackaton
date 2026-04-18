import sys
from pathlib import Path
import math

# 1. TRUCCO PER GLI IMPORT: 

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Ora Python capisce cosa significa "src" e gli import relativi funzioneranno!
from src.map_api import MapAPI
from src.governor_1 import runGovernor

def main():
    # 3. Percorso della mappa: ora che main.py è in src/, la mappa è nella sua stessa cartella
    current_dir = Path(__file__).resolve().parent
    csv_path = current_dir / "map_001_seed1.csv"
    
    if not csv_path.exists():
        print(f"Errore: Mappa non trovata in {csv_path}")
        return

    # 4. Parametri Missione
    start_point = (3.0, 3.0)
    target_point = (45.0, 45.0) 
    max_time = 1000000 #

    # 5. Inizializzazione API
    api = MapAPI(terrain=csv_path, rng_seed=42, time_step=0.2)

    print(f"Inizio missione: da {start_point} a {target_point}")
    
    # 6. Esecuzione del Governor
    results = runGovernor(api, start_point, target_point, max_time)

    # 7. Stampa Risultati
    (success, dist, time, d_dist, s_dist, r_dist, p_calls, s_calls, stuck) = results

    print("\n" + "="*30)
    print("      REPORT MISSIONE")
    print("="*30)
    print(f"Target Raggiunto:    {success}")
    print(f"Distanza Finale:     {dist:.2f} m")
    print(f"Tempo Impiegato:     {time:.2f} s")
    print(f"Viaggio Drone:       {d_dist:.2f} m")
    print(f"Chiamate Perceive:   {p_calls}")
    print(f"Chiamate Step:       {s_calls}")
    print(f"Eventi Stuck:        {stuck}")
    print("="*30)

if __name__ == "__main__":
    main()