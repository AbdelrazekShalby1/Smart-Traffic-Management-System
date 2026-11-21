from typing import Any, Dict, Optional


# ============================================================
# Runtime Context
# ============================================================
#
# This context will be connected to the existing CV system
# from main.py.
#
# The Agent does NOT create another detector, tracker, or
# VehicleStateManager.
#
# It uses the SAME live objects already running in the CV
# pipeline.
# ============================================================

_runtime_context: Dict[str, Any] = {
    "state_manager": None,
    "detector": None,
    "tracker": None,

    # Latest CV results
    "latest_tracks": None,
    "latest_density": None,
    "latest_count": None,
    "latest_speed": None,
    "latest_red_light": None,
    "update_count": 0,
}


def _log_tool_call(name: str):
    print(f"[AGENT] Calling tool: {name}")


# ============================================================
# Initialize / Connect CV System
# ============================================================

def set_traffic_context(
    state_manager=None,
    detector=None,
    tracker=None,
):
    """
    Connect the AI Agent to the existing Smart Traffic
    Computer Vision system.

    IMPORTANT:
    This function receives the already-created CV objects.

    It does NOT create new models or duplicate the CV pipeline.
    """

    _runtime_context["state_manager"] = state_manager
    _runtime_context["detector"] = detector
    _runtime_context["tracker"] = tracker

    print("[AGENT] Live traffic context connected")


# ============================================================
# Update Live CV Results
# ============================================================

def update_live_results(
    tracks=None,
    density=None,
    count=None,
    speed=None,
    red_light=None,
):
    """
    Store the latest results produced by the CV pipeline.

    main.py should call this after processing each frame.
    """

    if tracks is not None:
        _runtime_context["latest_tracks"] = tracks

    if density is not None:
        _runtime_context["latest_density"] = density

    if count is not None:
        _runtime_context["latest_count"] = count

    if speed is not None:
        _runtime_context["latest_speed"] = speed

    if red_light is not None:
        _runtime_context["latest_red_light"] = red_light

    _runtime_context["update_count"] += 1

    if _runtime_context["update_count"] == 1 or _runtime_context["update_count"] % 30 == 0:
        print("[AGENT] Live context updated")


# ============================================================
# Internal Helper
# ============================================================

def _get_state_manager():
    """
    Return the currently connected VehicleStateManager.
    """

    state_manager = _runtime_context.get("state_manager")

    if state_manager is None:
        raise RuntimeError(
            "Traffic CV system is not connected to the AI Agent. "
            "Call set_traffic_context() from main.py first."
        )

    return state_manager


def _active_vehicle_mapping() -> Dict[Any, Dict[str, Any]]:
    """
    Return active vehicles as an id -> vehicle mapping.
    """

    state_manager = _get_state_manager()
    vehicles = state_manager.get_active_vehicles()

    if isinstance(vehicles, dict):
        return vehicles

    return {
        vehicle.get("id", index): vehicle
        for index, vehicle in enumerate(vehicles)
        if isinstance(vehicle, dict)
    }


def _active_vehicle_list():
    """
    Return active vehicles as a list for tool responses.
    """

    return list(_active_vehicle_mapping().values())


# ============================================================
# Get Active Vehicles
# ============================================================

def get_active_vehicles() -> Dict[str, Any]:
    """
    Get all currently active vehicles from the existing
    VehicleStateManager.
    """

    try:
        _log_tool_call("get_active_vehicles")

        vehicles_by_id = _active_vehicle_mapping()
        vehicles = list(vehicles_by_id.values())

        return {
            "success": True,
            "count": len(vehicles),
            "vehicles": vehicles,
            "vehicles_by_id": vehicles_by_id,
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
            "vehicles": [],
        }


# ============================================================
# Get Vehicle Details
# ============================================================

def get_vehicle_details(
    vehicle_id: int,
) -> Dict[str, Any]:
    """
    Get detailed information about a currently tracked vehicle.
    """

    try:
        _log_tool_call("get_vehicle_details")
        state_manager = _get_state_manager()

        vehicle = state_manager.get_vehicle(vehicle_id)

        if vehicle is None:

            return {
                "success": False,
                "vehicle_id": vehicle_id,
                "error": (
                    f"Vehicle with tracking ID "
                    f"{vehicle_id} was not found."
                ),
            }

        return {
            "success": True,
            "vehicle": vehicle,
        }

    except Exception as e:

        return {
            "success": False,
            "vehicle_id": vehicle_id,
            "error": str(e),
        }


# ============================================================
# Get Live Traffic
# ============================================================

def get_live_traffic() -> Dict[str, Any]:
    """
    Get a snapshot of the current traffic situation.

    Data comes from the existing CV pipeline.
    """

    try:
        _log_tool_call("get_live_traffic")

        active_vehicles = _active_vehicle_list()

        # ----------------------------------------------------
        # Count vehicle types
        # ----------------------------------------------------

        vehicle_types: Dict[str, int] = {}

        for vehicle in active_vehicles:

            vehicle_type = vehicle.get(
                "type",
                "unknown",
            )

            vehicle_types[vehicle_type] = (
                vehicle_types.get(vehicle_type, 0) + 1
            )

        # ----------------------------------------------------
        # Speed information
        # ----------------------------------------------------

        speeds = []

        speeding_count = 0

        for vehicle in active_vehicles:

            speed = vehicle.get("speed")

            if speed is not None:

                speeds.append(float(speed))

            if vehicle.get("speeding", False):

                speeding_count += 1

        average_speed = None

        if speeds:

            average_speed = round(
                sum(speeds) / len(speeds),
                2,
            )

        # ----------------------------------------------------
        # Emergency vehicles
        # ----------------------------------------------------

        emergency_vehicles = [
            vehicle
            for vehicle in active_vehicles
            if vehicle.get("is_emergency", False)
        ]

        # ----------------------------------------------------
        # Density
        # ----------------------------------------------------

        density = _runtime_context.get(
            "latest_density"
        )

        return {
            "success": True,

            "active_vehicle_count": len(
                active_vehicles
            ),

            "vehicle_types": vehicle_types,

            "average_speed": average_speed,

            "speeding_vehicle_count": speeding_count,

            "emergency_vehicle_count": len(
                emergency_vehicles
            ),

            "density": density,
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
        }


# ============================================================
# Get Traffic Density
# ============================================================

def get_traffic_density() -> Dict[str, Any]:
    """
    Get the latest traffic density result produced by
    Traffic_Density.py.
    """

    density = _runtime_context.get(
        "latest_density"
    )

    if density is None:

        return {
            "success": False,
            "error": (
                "No traffic density result is currently "
                "available from the CV system."
            ),
        }

    _log_tool_call("get_traffic_density")

    return {
        "success": True,
        "density": density,
    }


# ============================================================
# Get Emergency Vehicles
# ============================================================

def get_emergency_vehicles() -> Dict[str, Any]:
    """
    Get currently detected emergency vehicles.

    Emergency information comes from the same CV tracking
    pipeline used by the main traffic system.
    """

    try:
        _log_tool_call("get_emergency_vehicles")

        active_vehicles = _active_vehicle_list()

        emergency_vehicles = [
            vehicle
            for vehicle in active_vehicles
            if vehicle.get("is_emergency", False)
        ]

        # ----------------------------------------------------
        # If VehicleStateManager does not contain emergency
        # state, use the latest tracker results as fallback.
        # ----------------------------------------------------

        if not emergency_vehicles:

            tracks = _runtime_context.get(
                "latest_tracks"
            )

            if tracks:

                emergency_tracks = tracks.get(
                    "emergency",
                    [],
                )

                emergency_vehicles = (
                    emergency_tracks
                )

        return {
            "success": True,
            "count": len(emergency_vehicles),
            "emergency_vehicles": emergency_vehicles,
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
            "emergency_vehicles": [],
        }


# ============================================================
# Get Current Traffic Count
# ============================================================

def get_current_vehicle_count() -> Dict[str, Any]:
    """
    Get the number of currently active vehicles.
    """

    try:
        _log_tool_call("get_current_vehicle_count")

        state_manager = _get_state_manager()

        count = state_manager.get_active_count()

        return {
            "success": True,
            "active_vehicle_count": count,
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
        }


# ============================================================
# Get Latest CV Results
# ============================================================

def get_latest_cv_results() -> Dict[str, Any]:
    """
    Return the latest results generated by the CV pipeline.

    Useful for debugging and advanced agent reasoning.
    """

    _log_tool_call("get_latest_cv_results")

    return {
        "success": True,

        "tracks": _runtime_context.get(
            "latest_tracks"
        ),

        "density": _runtime_context.get(
            "latest_density"
        ),

        "count": _runtime_context.get(
            "latest_count"
        ),

        "speed": _runtime_context.get(
            "latest_speed"
        ),

        "red_light": _runtime_context.get(
            "latest_red_light"
        ),
    }
