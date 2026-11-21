from langchain_core.tools import StructuredTool

from agent.tools.database_tools import (
    get_violations,
    get_statistics,
    get_violation_by_id,
    get_vehicle_violations,
)

from agent.tools.traffic_tools import (
    get_live_traffic,
    get_active_vehicles,
    get_vehicle_details,
    get_traffic_density,
    get_emergency_vehicles,
)

from agent.tools.analytics_tools import (
    get_speeding_vehicles,
    get_vehicle_count,
    get_average_speed,
    get_traffic_summary,
    analyze_traffic_situation,
)

from agent.tools.actions import (
    send_alert,
    generate_report,
    request_signal_change,
)


# ============================================================
# Smart Traffic Tools
# ============================================================

tools = [

    # --------------------------------------------------------
    # Live Traffic
    # --------------------------------------------------------

    StructuredTool.from_function(
        func=get_live_traffic,
        name="get_live_traffic",
        description=(
            "Get the current live traffic information from the "
            "Smart Traffic Computer Vision system. Use this for "
            "questions about current traffic conditions."
        ),
    ),

    StructuredTool.from_function(
        func=get_active_vehicles,
        name="get_active_vehicles",
        description=(
            "Get all currently active tracked vehicles detected "
            "by the Computer Vision system, including their IDs, "
            "types, positions, speeds, and states."
        ),
    ),

    StructuredTool.from_function(
        func=get_vehicle_details,
        name="get_vehicle_details",
        description=(
            "Get detailed information about a specific tracked "
            "vehicle using its tracking ID."
        ),
    ),

    StructuredTool.from_function(
        func=get_traffic_density,
        name="get_traffic_density",
        description=(
            "Get the current traffic density and congestion level "
            "detected by the Computer Vision system."
        ),
    ),

    StructuredTool.from_function(
        func=get_emergency_vehicles,
        name="get_emergency_vehicles",
        description=(
            "Get currently detected emergency vehicles such as "
            "ambulances, fire trucks, and police cars."
        ),
    ),

    # --------------------------------------------------------
    # Analytics
    # --------------------------------------------------------

    StructuredTool.from_function(
        func=get_speeding_vehicles,
        name="get_speeding_vehicles",
        description=(
            "Get vehicles that are currently exceeding the "
            "configured speed limit."
        ),
    ),

    StructuredTool.from_function(
        func=get_vehicle_count,
        name="get_vehicle_count",
        description=(
            "Get the current or recent vehicle count detected "
            "by the traffic monitoring system."
        ),
    ),

    StructuredTool.from_function(
        func=get_traffic_summary,
        name="get_traffic_summary",
        description=(
            "Get a summarized analysis of the current traffic "
            "situation including vehicles, density, speeding, "
            "emergency vehicles, and violations."
        ),
    ),

    StructuredTool.from_function(
        func=get_average_speed,
        name="get_average_speed",
        description=(
            "Get the average speed of currently active vehicles "
            "that have valid speed measurements."
        ),
    ),

    StructuredTool.from_function(
        func=analyze_traffic_situation,
        name="analyze_traffic_situation",
        description=(
            "Analyze the current traffic situation using measured "
            "CV data and simple rule-based observations."
        ),
    ),

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    StructuredTool.from_function(
        func=get_violations,
        name="get_violations",
        description=(
            "Get recorded traffic violations from the SQLite "
            "traffic database. Use this for historical or "
            "database-based violation questions."
        ),
    ),

    StructuredTool.from_function(
        func=get_statistics,
        name="get_statistics",
        description=(
            "Get traffic and violation statistics stored in "
            "the traffic database."
        ),
    ),

    StructuredTool.from_function(
        func=get_violation_by_id,
        name="get_violation_by_id",
        description=(
            "Get one recorded traffic violation by its database ID."
        ),
    ),

    StructuredTool.from_function(
        func=get_vehicle_violations,
        name="get_vehicle_violations",
        description=(
            "Get recorded violations associated with a specific "
            "tracked vehicle ID."
        ),
    ),

    # --------------------------------------------------------
    # Actions
    # --------------------------------------------------------

    StructuredTool.from_function(
        func=send_alert,
        name="send_alert",
        description=(
            "Send or prepare a traffic alert. Use this when "
            "the user explicitly asks to send an alert."
        ),
    ),

    StructuredTool.from_function(
        func=generate_report,
        name="generate_report",
        description=(
            "Generate a traffic report based on available "
            "traffic and violation data."
        ),
    ),

    StructuredTool.from_function(
        func=request_signal_change,
        name="request_signal_change",
        description=(
            "Request a traffic signal change. This is a "
            "consequential action and requires approval."
        ),
    ),
]
