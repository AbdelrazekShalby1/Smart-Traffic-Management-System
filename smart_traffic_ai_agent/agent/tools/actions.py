from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from config.settings import DB_PATH
from agent.tools.database_tools import (
    get_violations,
    get_statistics,
)


# ============================================================
# Send Alert
# ============================================================

def send_alert(
    alert_type: str,
    message: str,
    severity: str = "medium",
) -> Dict[str, Any]:
    """
    Prepare/send a traffic alert.

    Currently this function creates an alert payload and
    returns it.

    Later it can be connected to:
    - Email
    - Telegram
    - WhatsApp
    - SMS
    - Dashboard notifications
    - WebSocket
    - Mobile application
    """

    timestamp = datetime.now().isoformat()

    alert = {
        "alert_type": alert_type,
        "message": message,
        "severity": severity,
        "timestamp": timestamp,
        "status": "prepared",
    }

    return {
        "success": True,
        "alert": alert,
        "message": (
            "Alert has been prepared. "
            "No external notification service is currently "
            "connected."
        ),
    }


# ============================================================
# Generate Traffic Report
# ============================================================

def generate_report(
    report_type: str = "traffic_summary",
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Generate a traffic report using data from the existing
    Smart Traffic SQLite database.

    Available report types:

    - traffic_summary
    - violations
    - statistics
    """

    try:

        # ----------------------------------------------------
        # Statistics Report
        # ----------------------------------------------------

        if report_type == "statistics":

            statistics = get_statistics()

            if not statistics.get("success"):

                return statistics

            return {
                "success": True,
                "report_type": "statistics",
                "generated_at": datetime.now().isoformat(),
                "report": statistics.get(
                    "statistics",
                    {}
                ),
            }

        # ----------------------------------------------------
        # Violations Report
        # ----------------------------------------------------

        if report_type == "violations":

            violations = get_violations(
                limit=limit
            )

            if not violations.get("success"):

                return violations

            return {
                "success": True,
                "report_type": "violations",
                "generated_at": datetime.now().isoformat(),
                "report": {
                    "count": violations.get(
                        "count",
                        0
                    ),
                    "violations": violations.get(
                        "violations",
                        []
                    ),
                },
            }

        # ----------------------------------------------------
        # Traffic Summary
        # ----------------------------------------------------

        statistics = get_statistics()

        violations = get_violations(
            limit=limit
        )

        return {
            "success": True,
            "report_type": "traffic_summary",
            "generated_at": datetime.now().isoformat(),

            "report": {
                "statistics": statistics.get(
                    "statistics",
                    {}
                ),

                "recent_violations": violations.get(
                    "violations",
                    []
                ),

                "recent_violations_count": violations.get(
                    "count",
                    0
                ),
            },
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
        }


# ============================================================
# Request Traffic Signal Change
# ============================================================

def request_signal_change(
    intersection: str,
    requested_phase: str,
    reason: str,
) -> Dict[str, Any]:
    """
    Request a traffic signal change.

    IMPORTANT:
    This function DOES NOT directly change the traffic light.

    It creates an approval request because changing a real
    traffic signal is a consequential physical action.
    """

    timestamp = datetime.now().isoformat()

    request = {
        "intersection": intersection,
        "requested_phase": requested_phase,
        "reason": reason,
        "requested_at": timestamp,
        "status": "approval_required",
    }

    return {
        "success": True,

        "action": "traffic_signal_change",

        "status": "approval_required",

        "request": request,

        "message": (
            "Traffic signal change requested, "
            "but no signal was changed. "
            "Human approval is required before executing "
            "this action."
        ),
    }


# ============================================================
# Create Alert For Emergency Vehicle
# ============================================================

def create_emergency_alert(
    emergency_type: str,
    vehicle_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create an emergency traffic alert based on the CV system.
    """

    message = (
        f"Emergency vehicle detected: {emergency_type}."
    )

    if vehicle_id is not None:

        message += (
            f" Tracking ID: {vehicle_id}."
        )

    return send_alert(
        alert_type="EMERGENCY_VEHICLE",
        message=message,
        severity="high",
    )


# ============================================================
# Create Violation Alert
# ============================================================

def create_violation_alert(
    violation_type: str,
    vehicle_id: Optional[int] = None,
    details: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create an alert for a traffic violation.
    """

    message = (
        f"Traffic violation detected: "
        f"{violation_type}."
    )

    if vehicle_id is not None:

        message += (
            f" Vehicle ID: {vehicle_id}."
        )

    if details:

        message += (
            f" Details: {details}"
        )

    return send_alert(
        alert_type="TRAFFIC_VIOLATION",
        message=message,
        severity="medium",
    )