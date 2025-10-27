import cv2
import numpy as np
from typing import Any, List, Optional, Tuple
from webcam_osc.config import CellData, GridConfig


class DataVisualizer:
    def __init__(self, grid_config: GridConfig, show_camera: bool = True) -> None:
        self.grid_config: GridConfig = grid_config
        self.window_name: str = "Webcam OSC Visualizer"
        self.should_close: bool = False

        self.show_camera_runtime: bool = show_camera
        self.show_grid_runtime: bool = True

        self.max_height: int = 900
        self.max_width: int = 1600

        self.padding: int = 10
        self.text_padding: int = 5
        self.button_height: int = 35
        self.button_width: int = 140
        self.button_spacing: int = 10

        self._calculate_responsive_sizes()

        self._recalculate_layout()

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _calculate_responsive_sizes(self) -> None:
        desired_cell_size: int = 150
        desired_camera_width: int = 640
        desired_camera_height: int = 480

        desired_grid_width: int = (desired_cell_size * self.grid_config.cols) + (self.padding * (self.grid_config.cols + 1))
        desired_grid_height: int = (desired_cell_size * self.grid_config.rows) + (self.padding * (self.grid_config.rows + 1))

        total_desired_height: int = self.padding + desired_camera_height + self.padding + desired_grid_height + self.padding + self.button_height + self.padding

        scale_factor: float = 1.0
        if total_desired_height > self.max_height:
            scale_factor = self.max_height / total_desired_height

        self.cell_size: int = int(desired_cell_size * scale_factor)
        self.camera_width: int = int(desired_camera_width * scale_factor)
        self.camera_height: int = int(desired_camera_height * scale_factor)

        self.grid_width: int = (self.cell_size * self.grid_config.cols) + (self.padding * (self.grid_config.cols + 1))
        self.grid_height: int = (self.cell_size * self.grid_config.rows) + (self.padding * (self.grid_config.rows + 1))

    def _recalculate_layout(self) -> None:
        show_cam: bool = self.show_camera_runtime
        show_grid: bool = self.show_grid_runtime

        height: int = self.padding

        if show_cam:
            self.camera_y_offset: int = height
            height += self.camera_height + self.padding

        if show_grid:
            self.grid_y_offset: int = height
            height += self.grid_height + self.padding

        height += self.button_height + self.padding

        widths: List[int] = []
        if show_cam:
            widths.append(self.camera_width)
        if show_grid:
            widths.append(self.grid_width)

        content_width: int = max(widths) if widths else 400
        self.width: int = content_width + (2 * self.padding)
        self.height: int = height

        button_bar_y: int = self.height - self.button_height - self.padding
        total_button_width: int = (self.button_width * 3) + (self.button_spacing * 2)
        button_start_x: int = (self.width - total_button_width) // 2

        self.close_button_bounds: Tuple[int, int, int, int] = (
            button_start_x,
            button_bar_y,
            button_start_x + self.button_width,
            button_bar_y + self.button_height
        )

        self.toggle_camera_button_bounds: Tuple[int, int, int, int] = (
            button_start_x + self.button_width + self.button_spacing,
            button_bar_y,
            button_start_x + (2 * self.button_width) + self.button_spacing,
            button_bar_y + self.button_height
        )

        self.toggle_grid_button_bounds: Tuple[int, int, int, int] = (
            button_start_x + (2 * self.button_width) + (2 * self.button_spacing),
            button_bar_y,
            button_start_x + (3 * self.button_width) + (2 * self.button_spacing),
            button_bar_y + self.button_height
        )

    def render(self, cells_data: List[CellData], camera_frame: Optional[np.ndarray] = None) -> np.ndarray:
        canvas: np.ndarray = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas.fill(30)

        if self.show_camera_runtime and camera_frame is not None:
            resized_frame: np.ndarray = cv2.resize(camera_frame, (self.camera_width, self.camera_height))
            camera_x: int = (self.width - self.camera_width) // 2
            canvas[self.camera_y_offset:self.camera_y_offset + self.camera_height,
                   camera_x:camera_x + self.camera_width] = resized_frame

        if self.show_grid_runtime:
            grid_x_offset: int = (self.width - self.grid_width) // 2

            for cell_data in cells_data:
                x_offset: int = grid_x_offset + self.padding + (cell_data.col * (self.cell_size + self.padding))
                y_offset: int = self.grid_y_offset + self.padding + (cell_data.row * (self.cell_size + self.padding))

                r: int
                g: int
                b: int
                r, g, b = int(cell_data.dominant_color[0] * 255), int(cell_data.dominant_color[1] * 255), int(cell_data.dominant_color[2] * 255)
                cv2.rectangle(canvas,
                             (x_offset, y_offset),
                             (x_offset + self.cell_size, y_offset + self.cell_size),
                             (b, g, r),
                             -1)

                cv2.rectangle(canvas,
                             (x_offset, y_offset),
                             (x_offset + self.cell_size, y_offset + self.cell_size),
                             (100, 100, 100),
                             1)

                text_y: int = y_offset + 10
                font: int = cv2.FONT_HERSHEY_SIMPLEX
                font_scale: float = 0.35
                thickness: int = 1
                line_height: int = 14

                text_color: Tuple[int, int, int] = (50, 50, 50) if cell_data.brightness > 0.5 else (220, 220, 220)

                texts: List[str] = [
                    f"[{cell_data.row},{cell_data.col}]",
                    f"R:{cell_data.avg_red * 255:.0f}",
                    f"G:{cell_data.avg_green * 255:.0f}",
                    f"B:{cell_data.avg_blue * 255:.0f}",
                    f"Br:{cell_data.brightness:.2f}",
                    f"Cn:{cell_data.contrast:.2f}",
                    f"D:({r},{g},{b})"
                ]

                max_text_width: int = self.cell_size - (2 * self.text_padding)

                for text in texts:
                    if text_y + line_height > y_offset + self.cell_size - self.text_padding:
                        break

                    text_size: tuple[int, int] = cv2.getTextSize(text, font, font_scale, thickness)[0]  # type: ignore[assignment]
                    if text_size[0] > max_text_width:
                        while len(text) > 3 and cv2.getTextSize(text + "...", font, font_scale, thickness)[0][0] > max_text_width:
                            text = text[:-1]
                        text = text + "..."

                    cv2.putText(canvas, text,
                               (x_offset + self.text_padding, text_y),
                               font, font_scale, text_color, thickness, cv2.LINE_AA)
                    text_y += line_height

        self._draw_buttons(canvas)

        return canvas

    def _draw_buttons(self, canvas: np.ndarray) -> None:
        font: int = cv2.FONT_HERSHEY_SIMPLEX
        font_scale: float = 0.5
        thickness: int = 1

        self._draw_button(canvas, self.close_button_bounds, "Close",
                         (60, 60, 200), (80, 80, 220), font, font_scale, thickness)

        camera_text: str = "Hide Camera" if self.show_camera_runtime else "Show Camera"
        self._draw_button(canvas, self.toggle_camera_button_bounds, camera_text,
                         (60, 120, 60), (80, 150, 80), font, font_scale, thickness)

        grid_text: str = "Hide Grid" if self.show_grid_runtime else "Show Grid"
        self._draw_button(canvas, self.toggle_grid_button_bounds, grid_text,
                         (120, 60, 60), (150, 80, 80), font, font_scale, thickness)

    def _draw_button(self, canvas: np.ndarray, bounds: Tuple[int, int, int, int],
                     text: str, normal_color: Tuple[int, int, int],
                     hover_color: Tuple[int, int, int], font: int, font_scale: float, thickness: int) -> None:
        x1: int
        y1: int
        x2: int
        y2: int
        x1, y1, x2, y2 = bounds

        is_hovered: bool = self._is_mouse_over_bounds(bounds)
        button_color: Tuple[int, int, int] = hover_color if is_hovered else normal_color

        cv2.rectangle(canvas, (x1, y1), (x2, y2), button_color, -1)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (120, 120, 120), 1)

        text_size: tuple[int, int] = cv2.getTextSize(text, font, font_scale, thickness)[0]  # type: ignore[assignment]
        text_x: int = x1 + ((x2 - x1) - text_size[0]) // 2
        text_y: int = y1 + ((y2 - y1) + text_size[1]) // 2

        cv2.putText(canvas, text, (text_x, text_y),
                   font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def show_loading_screen(self, message: str = "Initializing...") -> None:
        loading_height: int = 300
        loading_width: int = 500
        canvas: np.ndarray = np.zeros((loading_height, loading_width, 3), dtype=np.uint8)
        canvas.fill(30)

        font: int = cv2.FONT_HERSHEY_SIMPLEX
        font_scale: float = 0.8
        thickness: int = 1

        text_size: tuple[int, int] = cv2.getTextSize(message, font, font_scale, thickness)[0]  # type: ignore[assignment]
        text_x: int = (loading_width - text_size[0]) // 2
        text_y: int = (loading_height + text_size[1]) // 2

        cv2.putText(canvas, message, (text_x, text_y),
                   font, font_scale, (200, 200, 200), thickness, cv2.LINE_AA)

        dots_y: int = text_y + 40
        i: int
        for i in range(3):
            dot_x: int = (loading_width // 2) - 30 + (i * 30)
            cv2.circle(canvas, (dot_x, dots_y), 5, (100, 100, 200), -1)

        try:
            cv2.imshow(self.window_name, canvas)
            cv2.waitKey(1)
        except cv2.error:
            pass

    def show(self, cells_data: List[CellData], camera_frame: Optional[np.ndarray] = None) -> None:
        canvas: np.ndarray = self.render(cells_data, camera_frame)
        try:
            cv2.imshow(self.window_name, canvas)
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                self.should_close = True
        except cv2.error:
            self.should_close = True

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: Optional[Any]) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x: int = x
            self.mouse_y: int = y
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self._is_point_in_bounds(x, y, self.close_button_bounds):
                self.should_close = True
            elif self._is_point_in_bounds(x, y, self.toggle_camera_button_bounds):
                self.show_camera_runtime = not self.show_camera_runtime
                self._recalculate_layout()
            elif self._is_point_in_bounds(x, y, self.toggle_grid_button_bounds):
                self.show_grid_runtime = not self.show_grid_runtime
                self._recalculate_layout()

    def _is_point_in_bounds(self, x: int, y: int, bounds: Tuple[int, int, int, int]) -> bool:
        x1: int
        y1: int
        x2: int
        y2: int
        x1, y1, x2, y2 = bounds
        return x1 <= x <= x2 and y1 <= y <= y2

    def _is_mouse_over_bounds(self, bounds: Tuple[int, int, int, int]) -> bool:
        if not hasattr(self, 'mouse_x') or not hasattr(self, 'mouse_y'):
            return False
        return self._is_point_in_bounds(self.mouse_x, self.mouse_y, bounds)

    def check_should_close(self) -> bool:
        return self.should_close

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)
