
import random

battery = 100.0

for i in range(10):
    drain = random.uniform(1.5, 3.5)
    battery = battery - drain
    
    print("Waypoint", i + 1)
    print("Current battery:", battery)
    
    if battery < 75.0:
        print("[EMERGENCY RETURN TRIGGERED]")
        break
        
    if battery < 80.0:
        print("Caution! Battery is getting low.")