import numpy as np
from typing import List, Tuple
from webcam_osc.config import CellData, GridConfig


class GridAnalyzer:
    def __init__(self, grid_config: GridConfig):
        self.grid_config = grid_config

    def analyze_frame(self, frame: np.ndarray) -> List[CellData]:
        height, width = frame.shape[:2]
        cell_height = height // self.grid_config.rows
        cell_width = width // self.grid_config.cols

        cells_data: List[CellData] = []

        for row in range(self.grid_config.rows):
            for col in range(self.grid_config.cols):
                y_start = row * cell_height
                y_end = (row + 1) * cell_height if row < self.grid_config.rows - 1 else height
                x_start = col * cell_width
                x_end = (col + 1) * cell_width if col < self.grid_config.cols - 1 else width

                cell = frame[y_start:y_end, x_start:x_end]
                cell_data = self._analyze_cell(cell, row, col)
                cells_data.append(cell_data)

        return cells_data

    def _analyze_cell(self, cell: np.ndarray, row: int, col: int) -> CellData:
        avg_color = np.mean(cell, axis=(0, 1))
        avg_blue, avg_green, avg_red = avg_color

        brightness = self._calculate_brightness(cell)
        contrast = self._calculate_contrast(cell)
        dominant = self._find_dominant_color(cell)

        return CellData(
            row=row,
            col=col,
            avg_red=float(avg_red / 255.0),
            avg_green=float(avg_green / 255.0),
            avg_blue=float(avg_blue / 255.0),
            brightness=brightness,
            contrast=contrast,
            dominant_color=dominant
        )

    def _calculate_brightness(self, cell: np.ndarray) -> float:
        gray = np.mean(cell)
        return float(gray / 255.0)

    def _calculate_contrast(self, cell: np.ndarray) -> float:
        gray = np.mean(cell, axis=2)
        std = np.std(gray)
        return float(std / 255.0)

    def _find_dominant_color(self, cell: np.ndarray) -> Tuple[float, float, float]:
        pixels = cell.reshape(-1, 3)
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        dominant_idx = np.argmax(counts)
        dominant_bgr = unique_colors[dominant_idx]
        return (float(dominant_bgr[2] / 255.0), float(dominant_bgr[1] / 255.0), float(dominant_bgr[0] / 255.0))
