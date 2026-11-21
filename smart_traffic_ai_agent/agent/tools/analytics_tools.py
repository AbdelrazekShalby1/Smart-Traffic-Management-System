from typing import Any, Dict, List

from agent.tools.traffic_tools import (
    get_active_vehicles,
    get_live_traffic,
    _log_tool_call,
)


# ============================================================
# Get Speeding Vehicles
# ============================================================

def get_speeding_vehicles() -> Dict[str, Any]:
    """
    Get vehicles that are currently exceeding the configured
    speed limit.

    The speed information comes from VehicleStateManager,
    which is updated by Speed_Estimation.py.
    """

    try:
        _log_tool_call("get_speeding_vehicles")

        result = get_active_vehicles()

        if not result.get("success"):
            return result

        vehicles = result.get("vehicles", [])

        speeding_vehicles = []

        for vehicle in vehicles:

            if vehicle.get("speeding", False):

                speeding_vehicles.append({
                    "id": vehicle.get("id"),
                    "type": vehicle.get("type"),
                    "speed": vehicle.get("speed"),
                    "bbox": vehicle.get("bbox"),
                    "violations": vehicle.get(
                        "violations",
                        []
                    ),
                })

        return {
            "success": True,
            "count": len(speeding_vehicles),
            "speeding_vehicles": speeding_vehicles,
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
            "speeding_vehicles": [],
        }


# ============================================================
# Get Vehicle Count
# ============================================================

def get_vehicle_count() -> Dict[str, Any]:
    """
    Get the current number of active vehicles and a breakdown
    by vehicle type.
    """

    try:
        _log_tool_call("get_vehicle_count")

        result = get_active_vehicles()

        if not result.get("success"):
            return result

        vehicles = result.get(
            "vehicles",
            []
        )

        vehicle_types: Dict[str, int] = {}

        for vehicle in vehicles:

            vehicle_type = vehicle.get(
                "type",
                "unknown"
            )

            vehicle_types[vehicle_type] = (
                vehicle_types.get(
                    vehicle_type,
                    0
                ) + 1
            )

        return {
            "success": True,
            "total_active_vehicles": len(
                vehicles
            ),
            "vehicle_types": vehicle_types,
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
        }


# ============================================================
# Get Average Speed
# ============================================================

def get_average_speed() -> Dict[str, Any]:
    """
    Calculate the average speed of currently active vehicles
    that have a valid speed measurement.
    """

    try:
        _log_tool_call("get_average_speed")

        result = get_active_vehicles()

        if not result.get("success"):
            return result

        vehicles = result.get(
            "vehicles",
            []
        )

        speeds: List[float] = []

        for vehicle in vehicles:

            speed = vehicle.get("speed")

            if speed is None:
                continue

            try:
                speeds.append(
                    float(speed)
                )
            except (TypeError, ValueError):
                continue

        if not speeds:

            return {
                "success": True,
                "average_speed": None,
                "measured_vehicles": 0,
            }

        average_speed = (
            sum(speeds) / len(speeds)
        )

        return {
            "success": True,
            "average_speed": round(
                average_speed,
                2
            ),
            "measured_vehicles": len(
                speeds
            ),
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
        }


# ============================================================
# Get Traffic Summary
# ============================================================

def get_traffic_summary() -> Dict[str, Any]:
    """
    Generate a real-time summary of the current traffic
    situation.

    Combines:
    - Active vehicles
    - Vehicle types
    - Average speed
    - Speeding vehicles
    - Emergency vehicles
    - Traffic density
    """

    try:
        _log_tool_call("get_traffic_summary")

        traffic = get_live_traffic()

        if not traffic.get("success"):

            return traffic

        speeding = get_speeding_vehicles()

        vehicle_count = get_vehicle_count()

        average_speed = get_average_speed()

        return {
            "success": True,

            "traffic": {
                "active_vehicle_count": traffic.get(
                    "active_vehicle_count"
                ),

                "vehicle_types": traffic.get(
                    "vehicle_types",
                    {}
                ),

                "average_speed": average_speed.get(
                    "average_speed"
                ),

                "speeding_vehicle_count": speeding.get(
                    "count",
                    0
                ),

                "emergency_vehicle_count": traffic.get(
                    "emergency_vehicle_count",
                    0
                ),

                "density": traffic.get(
                    "density"
                ),

                "vehicle_count_by_type": (
                    vehicle_count.get(
                        "vehicle_types",
                        {}
                    )
                ),
            }
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
        }


# ============================================================
# Analyze Traffic Situation
# ============================================================

def analyze_traffic_situation() -> Dict[str, Any]:
    """
    Produce a basic rule-based interpretation of the current
    traffic state.

    This is NOT an LLM prediction.

    It only interprets measurements already produced by the
    CV system.
    """

    try:
        _log_tool_call("analyze_traffic_situation")

        summary = get_traffic_summary()

        if not summary.get("success"):
            return summary

        traffic = summary.get(
            "traffic",
            {}
        )

        active_count = traffic.get(
            "active_vehicle_count",
            0
        )

        speeding_count = traffic.get(
            "speeding_vehicle_count",
            0
        )

        emergency_count = traffic.get(
            "emergency_vehicle_count",
            0
        )

        density = traffic.get(
            "density"
        )

        observations = []

        # ----------------------------------------------------
        # Vehicle observations
        # ----------------------------------------------------

        if active_count == 0:

            observations.append(
                "No active vehicles are currently available."
            )

        else:

            observations.append(
                f"{active_count} active vehicles "
                "are currently being tracked."
            )

        # ----------------------------------------------------
        # Speeding
        # ----------------------------------------------------

        if speeding_count > 0:

            observations.append(
                f"{speeding_count} vehicles are "
                "currently exceeding the speed limit."
            )

        # ----------------------------------------------------
        # Emergency vehicles
        # ----------------------------------------------------

        if emergency_count > 0:

            observations.append(
                f"{emergency_count} emergency "
                "vehicle(s) are currently detected."
            )

        # ----------------------------------------------------
        # Density
        # ----------------------------------------------------

        if density is not None:

            observations.append(
                f"Current traffic density result: "
                f"{density}."
            )

        return {
            "success": True,

            "observations": observations,

            "measurements": traffic,

            "note": (
                "These observations are based on "
                "current CV measurements and are "
                "not predictions."
            ),
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
        }
