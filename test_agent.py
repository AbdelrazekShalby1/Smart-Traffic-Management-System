import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent / "smart_traffic_ai_agent"

sys.path.insert(0, str(AGENT_ROOT))

from agent.tools.traffic_tools import (
    get_live_traffic,
    get_active_vehicles,
    get_traffic_density,
    get_emergency_vehicles,
)

from agent.tools.analytics_tools import (
    get_speeding_vehicles,
    get_vehicle_count,
    get_traffic_summary,
)

from agent.tools.database_tools import (
    get_violations,
    get_statistics,
)


print("=" * 60)
print("TESTING TRAFFIC TOOLS")
print("=" * 60)


print("\n1. Live Traffic:")
try:
    print(get_live_traffic())
except Exception as e:
    print("ERROR:", e)


print("\n2. Active Vehicles:")
try:
    print(get_active_vehicles())
except Exception as e:
    print("ERROR:", e)


print("\n3. Traffic Density:")
try:
    print(get_traffic_density())
except Exception as e:
    print("ERROR:", e)


print("\n4. Emergency Vehicles:")
try:
    print(get_emergency_vehicles())
except Exception as e:
    print("ERROR:", e)


print("\n5. Speeding Vehicles:")
try:
    print(get_speeding_vehicles())
except Exception as e:
    print("ERROR:", e)


print("\n6. Vehicle Count:")
try:
    print(get_vehicle_count())
except Exception as e:
    print("ERROR:", e)


print("\n7. Traffic Summary:")
try:
    print(get_traffic_summary())
except Exception as e:
    print("ERROR:", e)


print("\n8. Database Violations:")
try:
    print(get_violations())
except Exception as e:
    print("ERROR:", e)


print("\n9. Database Statistics:")
try:
    print(get_statistics())
except Exception as e:
    print("ERROR:", e)


print("\n" + "=" * 60)
print("TEST FINISHED")
print("=" * 60)