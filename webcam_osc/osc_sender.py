from pythonosc import udp_client
from pythonosc.osc_bundle_builder import OscBundleBuilder
from pythonosc.osc_message_builder import OscMessageBuilder
from typing import List
from webcam_osc.config import CellData, OSCConfig


class OSCSender:
    def __init__(self, osc_config: OSCConfig) -> None:
        self.client: udp_client.SimpleUDPClient = udp_client.SimpleUDPClient(osc_config.host, osc_config.port)

    def send_grid_data(self, cells: List[CellData]) -> None:
        bundle: OscBundleBuilder = OscBundleBuilder(0)

        for cell in cells:
            address: str = f"/cell/{cell.row}/{cell.col}"

            rgb_builder: OscMessageBuilder = OscMessageBuilder(address=f"{address}/rgb")
            rgb_builder.add_arg(cell.avg_red)
            rgb_builder.add_arg(cell.avg_green)
            rgb_builder.add_arg(cell.avg_blue)
            bundle.add_content(rgb_builder.build())  # type: ignore[arg-type]

            brightness_builder: OscMessageBuilder = OscMessageBuilder(address=f"{address}/brightness")
            brightness_builder.add_arg(cell.brightness)
            bundle.add_content(brightness_builder.build())  # type: ignore[arg-type]

            contrast_builder: OscMessageBuilder = OscMessageBuilder(address=f"{address}/contrast")
            contrast_builder.add_arg(cell.contrast)
            bundle.add_content(contrast_builder.build())  # type: ignore[arg-type]

            dominant_builder: OscMessageBuilder = OscMessageBuilder(address=f"{address}/dominant")
            dominant_builder.add_arg(cell.dominant_color[0])
            dominant_builder.add_arg(cell.dominant_color[1])
            dominant_builder.add_arg(cell.dominant_color[2])
            bundle.add_content(dominant_builder.build())  # type: ignore[arg-type]

        self.client.send(bundle.build())
