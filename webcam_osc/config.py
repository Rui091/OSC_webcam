from dataclasses import dataclass
from typing import Tuple


@dataclass
class GridConfig:
    rows: int
    cols: int


@dataclass
class OSCConfig:
    host: str = "127.0.0.1"
    port: int = 5005


@dataclass
class CellData:
    row: int
    col: int
    avg_red: float
    avg_green: float
    avg_blue: float
    brightness: float
    contrast: float
    dominant_color: Tuple[float, float, float]


@dataclass
class AppConfig:
    grid: GridConfig
    osc: OSCConfig
    camera_index: int = 0
    target_fps: int = 30
    show_visualizer: bool = True
    show_camera: bool = True
