"""
Telemetry schema definitions for ROS2/NVIDIA DRIVE fleet data.

Reflects ROS2 message type conventions: sensor_msgs/Imu, sensor_msgs/PointCloud2,
automotive CAN bus signals, and perception pipeline metadata from Waymo/Brembo-style
autonomous vehicle stacks.
"""

from typing import Dict, List, Any

# ROS2-style sensor message types
IMU_SCHEMA: Dict[str, str] = {
    "timestamp_ns": "int64",
    "vehicle_id": "str",
    "sensor_type": "str",  # sensor_msgs/Imu
    "accel_x": "float64",
    "accel_y": "float64",
    "accel_z": "float64",
    "gyro_x": "float64",
    "gyro_y": "float64",
    "gyro_z": "float64",
    "orientation_w": "float64",
    "orientation_x": "float64",
    "orientation_y": "float64",
    "orientation_z": "float64",
}

LIDAR_SCHEMA: Dict[str, str] = {
    "timestamp_ns": "int64",
    "vehicle_id": "str",
    "sensor_type": "str",  # sensor_msgs/PointCloud2
    "point_count": "int64",
    "min_range": "float64",
    "max_range": "float64",
    "mean_intensity": "float64",
    "frame_id": "str",
}

CAN_SCHEMA: Dict[str, str] = {
    "timestamp_ns": "int64",
    "vehicle_id": "str",
    "sensor_type": "str",  # CAN/Ethernet virtual sensors
    "vehicle_speed_kmh": "float64",
    "brake_pressure_pct": "float64",
    "steering_angle_deg": "float64",
    "throttle_position_pct": "float64",
    "engine_rpm": "float64",
    "gear_position": "int64",
}

GPS_SCHEMA: Dict[str, str] = {
    "timestamp_ns": "int64",
    "vehicle_id": "str",
    "sensor_type": "str",  # nav_msgs/Odometry or sensor_msgs/NavSatFix
    "latitude": "float64",
    "longitude": "float64",
    "altitude_m": "float64",
    "velocity_north": "float64",
    "velocity_east": "float64",
}

CAMERA_SCHEMA: Dict[str, str] = {
    "timestamp_ns": "int64",
    "vehicle_id": "str",
    "sensor_type": "str",  # sensor_msgs/Image metadata
    "frame_id": "str",
    "exposure_ms": "float64",
    "object_count": "int64",
    "resolution_w": "int64",
    "resolution_h": "int64",
}

SENSOR_TYPES: List[str] = ["imu", "lidar", "can", "gps", "camera"]

# Unified schema: all columns from all sensor types (flattened for Parquet)
# Common columns plus sensor-specific columns with nullable extras
TELEMETRY_SCHEMA: Dict[str, str] = {
    "timestamp_ns": "int64",
    "vehicle_id": "str",
    "sensor_type": "str",
    "accel_x": "float64",
    "accel_y": "float64",
    "accel_z": "float64",
    "gyro_x": "float64",
    "gyro_y": "float64",
    "gyro_z": "float64",
    "orientation_w": "float64",
    "orientation_x": "float64",
    "orientation_y": "float64",
    "orientation_z": "float64",
    "point_count": "int64",
    "min_range": "float64",
    "max_range": "float64",
    "mean_intensity": "float64",
    "vehicle_speed_kmh": "float64",
    "brake_pressure_pct": "float64",
    "steering_angle_deg": "float64",
    "throttle_position_pct": "float64",
    "engine_rpm": "float64",
    "gear_position": "int64",
    "latitude": "float64",
    "longitude": "float64",
    "altitude_m": "float64",
    "velocity_north": "float64",
    "velocity_east": "float64",
    "frame_id": "str",
    "exposure_ms": "float64",
    "object_count": "int64",
    "resolution_w": "int64",
    "resolution_h": "int64",
}


def get_columns_for_sensor(sensor_type: str) -> List[str]:
    """Return required columns for a given sensor type."""
    schemas = {
        "imu": list(IMU_SCHEMA.keys()),
        "lidar": list(LIDAR_SCHEMA.keys()),
        "can": list(CAN_SCHEMA.keys()),
        "gps": list(GPS_SCHEMA.keys()),
        "camera": list(CAMERA_SCHEMA.keys()),
    }
    return schemas.get(sensor_type, list(TELEMETRY_SCHEMA.keys()))


def validate_schema(data: Any, expected: Dict[str, str]) -> bool:
    """Validate that data has expected schema. Returns True if valid."""
    if hasattr(data, "columns"):
        cols = set(data.columns)
    elif isinstance(data, dict):
        cols = set(data.keys())
    else:
        return False
    return set(expected.keys()).issubset(cols)
