import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from config.settings import DB_PATH


def _log_tool_call(name: str):
    print(f"[AGENT] Calling tool: {name}")


# ============================================================
# Database Connection
# ============================================================

def _get_connection() -> sqlite3.Connection:
    """
    Create a connection to the Smart Traffic SQLite database.
    """

    db_path = Path(DB_PATH).resolve()

    if not db_path.exists():
        raise sqlite3.OperationalError(
            f"Traffic database not found at {db_path}. "
            "Check TRAFFIC_DB_PATH in project_root/.env."
        )

    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        AND name = 'violations'
        """
    )

    if cursor.fetchone() is None:
        connection.close()
        raise sqlite3.OperationalError(
            "The violations table does not exist in "
            f"{db_path}."
        )

    return connection


# ============================================================
# Get Violations
# ============================================================

def get_violations(limit: int = 20) -> Dict[str, Any]:
    """
    Get recorded traffic violations from the database.

    Parameters
    ----------
    limit : int
        Maximum number of violations to return.

    Returns
    -------
    dict
        Violations and metadata.
    """

    if limit <= 0:
        limit = 20

    try:
        _log_tool_call("get_violations")
        connection = _get_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                vehicle_id,
                vehicle_type,
                violation_type,
                speed,
                license_plate,
                timestamp,
                evidence_path
            FROM violations
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )

        rows = cursor.fetchall()

        connection.close()

        violations: List[Dict[str, Any]] = [
            dict(row) for row in rows
        ]

        return {
            "success": True,
            "count": len(violations),
            "violations": violations,
        }

    except sqlite3.Error as e:

        return {
            "success": False,
            "error": f"Database error: {str(e)}",
            "violations": [],
        }


# ============================================================
# Get Statistics
# ============================================================

def get_statistics() -> Dict[str, Any]:
    """
    Get traffic violation statistics from the database.

    Returns statistics such as:
    - Total violations
    - Speeding violations
    - Red-light violations
    - Average recorded speed
    """

    try:
        _log_tool_call("get_statistics")
        connection = _get_connection()

        cursor = connection.cursor()

        # ----------------------------------------------------
        # Total violations
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM violations
            """
        )

        total_violations = cursor.fetchone()[0]

        # ----------------------------------------------------
        # Speeding violations
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM violations
            WHERE UPPER(violation_type) = 'SPEEDING'
            """
        )

        speeding_violations = cursor.fetchone()[0]

        # ----------------------------------------------------
        # Red-light violations
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM violations
            WHERE
                UPPER(violation_type) IN (
                    'RED_LIGHT',
                    'RED LIGHT',
                    'RED_LIGHT_VIOLATION',
                    'RED LIGHT VIOLATION'
                )
            """
        )

        red_light_violations = cursor.fetchone()[0]

        # ----------------------------------------------------
        # Average speed
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT AVG(speed)
            FROM violations
            WHERE speed IS NOT NULL
            """
        )

        average_speed = cursor.fetchone()[0]

        # ----------------------------------------------------
        # Vehicle types
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                vehicle_type,
                COUNT(*) AS count
            FROM violations
            GROUP BY vehicle_type
            ORDER BY count DESC
            """
        )

        vehicle_types = {
            row["vehicle_type"]: row["count"]
            for row in cursor.fetchall()
        }

        # ----------------------------------------------------
        # Violation types
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                violation_type,
                COUNT(*) AS count
            FROM violations
            GROUP BY violation_type
            ORDER BY count DESC
            """
        )

        violation_types = {
            row["violation_type"]: row["count"]
            for row in cursor.fetchall()
        }

        connection.close()

        return {
            "success": True,
            "statistics": {
                "total_violations": total_violations,
                "speeding_violations": speeding_violations,
                "red_light_violations": red_light_violations,
                "average_recorded_speed": (
                    round(float(average_speed), 2)
                    if average_speed is not None
                    else None
                ),
                "vehicle_types": vehicle_types,
                "violation_types": violation_types,
            },
        }

    except sqlite3.Error as e:

        return {
            "success": False,
            "error": f"Database error: {str(e)}",
            "statistics": {},
        }


# ============================================================
# Get Violation By ID
# ============================================================

def get_violation_by_id(
    violation_id: int,
) -> Dict[str, Any]:
    """
    Get a specific violation using its database ID.
    """

    try:
        _log_tool_call("get_violation_by_id")
        connection = _get_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                vehicle_id,
                vehicle_type,
                violation_type,
                speed,
                license_plate,
                timestamp,
                evidence_path
            FROM violations
            WHERE id = ?
            """,
            (violation_id,),
        )

        row = cursor.fetchone()

        connection.close()

        if row is None:
            return {
                "success": False,
                "error": f"Violation {violation_id} was not found.",
                "violation": None,
            }

        return {
            "success": True,
            "violation": dict(row),
        }

    except sqlite3.Error as e:

        return {
            "success": False,
            "error": f"Database error: {str(e)}",
            "violation": None,
        }


# ============================================================
# Get Vehicle Violations
# ============================================================

def get_vehicle_violations(
    vehicle_id: int,
) -> Dict[str, Any]:
    """
    Get all recorded violations associated with a specific
    tracked vehicle ID.
    """

    try:
        _log_tool_call("get_vehicle_violations")
        connection = _get_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                vehicle_id,
                vehicle_type,
                violation_type,
                speed,
                license_plate,
                timestamp,
                evidence_path
            FROM violations
            WHERE vehicle_id = ?
            ORDER BY id DESC
            """,
            (vehicle_id,),
        )

        rows = cursor.fetchall()

        connection.close()

        violations = [
            dict(row) for row in rows
        ]

        return {
            "success": True,
            "vehicle_id": vehicle_id,
            "count": len(violations),
            "violations": violations,
        }

    except sqlite3.Error as e:

        return {
            "success": False,
            "error": f"Database error: {str(e)}",
            "vehicle_id": vehicle_id,
            "violations": [],
        }
