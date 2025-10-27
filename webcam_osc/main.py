import time
import cv2
from typing import Optional
from webcam_osc.config import GridConfig, OSCConfig, AppConfig
from webcam_osc.capture import WebcamCapture
from webcam_osc.analyzer import GridAnalyzer
from webcam_osc.osc_sender import OSCSender
from webcam_osc.visualizer import DataVisualizer


def main() -> None:
    config: AppConfig = AppConfig(
        grid=GridConfig(rows=4, cols=4),
        osc=OSCConfig(host="127.0.0.1", port=5005),
        camera_index=0,
        target_fps=30
    )

    visualizer: Optional[DataVisualizer] = DataVisualizer(config.grid, config.show_camera) if config.show_visualizer else None

    if visualizer:
        visualizer.show_loading_screen("Initializing components...")

    analyzer: GridAnalyzer = GridAnalyzer(config.grid)

    if visualizer:
        visualizer.show_loading_screen("Connecting to OSC...")

    osc_sender: OSCSender = OSCSender(config.osc)

    frame_delay: float = 1.0 / config.target_fps

    if visualizer:
        visualizer.show_loading_screen("Starting camera...")

    with WebcamCapture(config.camera_index) as capture:
        if not capture.start():
            print("Failed to open camera")
            return

        if visualizer:
            visualizer.show_loading_screen("Ready! Starting stream...")
            time.sleep(0.5)

        print(f"Streaming {config.grid.rows}x{config.grid.cols} grid to {config.osc.host}:{config.osc.port}")
        if config.show_visualizer:
            print(f"Data visualizer enabled (camera: {'on' if config.show_camera else 'off'})")
        elif config.show_camera:
            print("Camera preview enabled")

        while True:
            start_time = time.time()

            frame = capture.get_frame()
            if frame is None:
                break

            cells_data = analyzer.analyze_frame(frame)
            osc_sender.send_grid_data(cells_data)

            if config.show_camera and not config.show_visualizer:
                cv2.imshow("Webcam Grid", frame)

            if visualizer:
                visualizer.show(cells_data, frame if config.show_camera else None)
                if visualizer.check_should_close():
                    print("\nClosing application...")
                    break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elapsed = time.time() - start_time
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)

        cv2.destroyAllWindows()
        if visualizer:
            visualizer.close()


if __name__ == "__main__":
    main()
