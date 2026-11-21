import streamlit as st
import sqlite3
import pandas as pd
import os


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Traffic Management System",
    page_icon="🚦",
    layout="wide"
)


# =========================================================
# DATABASE CONFIG
# =========================================================

DB_PATH = os.path.join(
    "DataBase",
    "traffic_system.db"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 20px;
        color: #777;
        margin-bottom: 25px;
    }

    .violation-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DATABASE
# =========================================================

def load_data():

    if not os.path.exists(DB_PATH):

        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)

    query = """
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
    """

    try:

        df = pd.read_sql_query(
            query,
            conn
        )

    except Exception:

        df = pd.DataFrame()

    conn.close()

    return df


# =========================================================
# HEADER
# =========================================================

col1, col2 = st.columns(
    [5, 1]
)


with col1:

    st.markdown(
        '<div class="main-title">🚦 Smart Traffic Management System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Computer Vision Traffic Monitoring Dashboard</div>',
        unsafe_allow_html=True
    )


with col2:

    if st.button(
        "🔄 Refresh",
        use_container_width=True
    ):

        st.rerun()


# =========================================================
# LOAD DATA
# =========================================================

df = load_data()


if df.empty:

    st.warning(
        "⚠️ No violations found in the database."
    )

    st.info(
        f"Database path: `{DB_PATH}`"
    )

    st.stop()


# =========================================================
# DATA PREPARATION
# =========================================================

df["speed"] = pd.to_numeric(
    df["speed"],
    errors="coerce"
)

df["license_plate"] = (
    df["license_plate"]
    .fillna("Not Detected")
)

df["vehicle_type"] = (
    df["vehicle_type"]
    .fillna("Unknown")
)

df["violation_type"] = (
    df["violation_type"]
    .fillna("Unknown")
)


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.title("🎛️ Filters")


# ---------------------------------------------------------
# Violation Filter
# ---------------------------------------------------------

violation_options = sorted(
    df["violation_type"]
    .unique()
    .tolist()
)

selected_violations = st.sidebar.multiselect(
    "Violation Type",
    violation_options,
    default=violation_options
)


# ---------------------------------------------------------
# Vehicle Filter
# ---------------------------------------------------------

vehicle_options = sorted(
    df["vehicle_type"]
    .unique()
    .tolist()
)

selected_vehicles = st.sidebar.multiselect(
    "Vehicle Type",
    vehicle_options,
    default=vehicle_options
)


# ---------------------------------------------------------
# License Plate Filter
# ---------------------------------------------------------

plate_search = st.sidebar.text_input(
    "🔎 License Plate",
    ""
)


# =========================================================
# APPLY FILTERS
# =========================================================

filtered_df = df[
    df["violation_type"].isin(
        selected_violations
    )
    &
    df["vehicle_type"].isin(
        selected_vehicles
    )
].copy()


if plate_search:

    filtered_df = filtered_df[
        filtered_df["license_plate"]
        .str.contains(
            plate_search,
            case=False,
            na=False
        )
    ]


# =========================================================
# METRICS
# =========================================================

st.divider()

total_violations = len(
    filtered_df
)


speeding_count = len(
    filtered_df[
        filtered_df["violation_type"] == "SPEEDING"
    ]
)


red_light_count = len(
    filtered_df[
        filtered_df["violation_type"] == "RED_LIGHT"
    ]
)


vehicles_count = (
    filtered_df["vehicle_id"]
    .nunique()
)


average_speed = (
    filtered_df["speed"].mean()
    if not filtered_df.empty
    else 0
)


col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "🚨 Total Violations",
        total_violations
    )


with col2:

    st.metric(
        "⚡ Speeding",
        speeding_count
    )


with col3:

    st.metric(
        "🚦 Red Light",
        red_light_count
    )


with col4:

    st.metric(
        "🚗 Vehicles",
        vehicles_count
    )


with col5:

    st.metric(
        "📊 Avg Speed",
        f"{average_speed:.1f} km/h"
    )


# =========================================================
# NO RESULTS
# =========================================================

if filtered_df.empty:

    st.warning(
        "No violations match the selected filters."
    )

    st.stop()


# =========================================================
# CHARTS
# =========================================================

st.divider()

st.subheader("📊 Traffic Analytics")


col1, col2 = st.columns(2)


# ---------------------------------------------------------
# Violation Types
# ---------------------------------------------------------

with col1:

    st.markdown("### 🚨 Violations by Type")

    violation_counts = (
        filtered_df["violation_type"]
        .value_counts()
    )

    st.bar_chart(
        violation_counts
    )


# ---------------------------------------------------------
# Vehicle Types
# ---------------------------------------------------------

with col2:

    st.markdown("### 🚗 Violations by Vehicle")

    vehicle_counts = (
        filtered_df["vehicle_type"]
        .value_counts()
    )

    st.bar_chart(
        vehicle_counts
    )


# =========================================================
# SPEED ANALYSIS
# =========================================================

st.divider()

st.subheader("⚡ Speed Analysis")


speed_data = filtered_df[
    filtered_df["speed"].notna()
][
    ["vehicle_id", "speed"]
]


if not speed_data.empty:

    st.line_chart(
        speed_data.set_index(
            "vehicle_id"
        )
    )

else:

    st.info(
        "No speed data available."
    )


# =========================================================
# VIOLATIONS TABLE
# =========================================================

st.divider()

st.subheader("📋 Violation Records")


display_df = filtered_df[
    [
        "id",
        "vehicle_id",
        "vehicle_type",
        "violation_type",
        "speed",
        "license_plate",
        "timestamp"
    ]
].copy()


display_df.columns = [
    "ID",
    "Vehicle ID",
    "Vehicle Type",
    "Violation",
    "Speed (km/h)",
    "License Plate",
    "Timestamp"
]


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# EVIDENCE
# =========================================================

st.divider()

st.subheader("🖼️ Violation Evidence")


selected_id = st.selectbox(
    "Select Violation",
    filtered_df["id"].tolist()
)


selected_row = filtered_df[
    filtered_df["id"] == selected_id
].iloc[0]


# =========================================================
# EVIDENCE DETAILS
# =========================================================

col1, col2 = st.columns(
    [2, 1]
)


# ---------------------------------------------------------
# IMAGE
# ---------------------------------------------------------

with col1:

    evidence_path = selected_row[
        "evidence_path"
    ]

    # -----------------------------------------------------
    # Try path directly
    # -----------------------------------------------------

    possible_paths = [
        evidence_path,
        os.path.join(
            os.getcwd(),
            evidence_path
        )
    ]

    actual_path = None

    for path in possible_paths:

        if path and os.path.exists(path):

            actual_path = path

            break


    if actual_path:

        st.image(
            actual_path,
            caption=(
                f"Violation #{selected_row['id']}"
            ),
            use_container_width=True
        )

    else:

        st.error(
            "❌ Evidence image not found."
        )

        st.code(
            str(evidence_path)
        )


# ---------------------------------------------------------
# DETAILS
# ---------------------------------------------------------

with col2:

    st.markdown("### 🚨 Violation Details")

    st.markdown(
        f"""
        **Violation ID:**  
        `{selected_row['id']}`

        **Vehicle ID:**  
        `{selected_row['vehicle_id']}`

        **Vehicle Type:**  
        `{selected_row['vehicle_type']}`

        **Violation:**  
        `{selected_row['violation_type']}`

        **Speed:**  
        `{selected_row['speed']:.1f} km/h`
        """
    )

    license_plate = selected_row[
        "license_plate"
    ]

    if license_plate == "Not Detected":

        st.warning(
            "🔍 License Plate: Not Detected"
        )

    else:

        st.success(
            f"🔖 License Plate: {license_plate}"
        )

    st.markdown(
        f"""
        **Timestamp:**  
        `{selected_row['timestamp']}`
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🚦 Smart Traffic Management System | "
    "Computer Vision • Object Tracking • "
    "Speed Detection • Red Light Detection • "
    "License Plate Recognition"
)