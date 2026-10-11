import json
import socket
from datetime import datetime, timezone


class V2XBroadcaster:
    def __init__(self, host: str = "255.255.255.255", port: int = 37020) -> None:
        self.address = (host, port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def send_intent_alert(self, intent_label: str, confidence: float, lookahead_sec: float = 2.5, predicted_status: str = "ALERT") -> dict[str, object]:
        status_command = {
            "DROWSY": "DRIVER_ALERT_RIDER_DROWSY",
            "DISTRACTED": "DRIVER_ALERT_RIDER_DISTRACTED",
        }.get(predicted_status, "DRIVER_STATUS_ALERT")
        payload = {
            "node_type": "GARUDA_EDGE_SYSTEM",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "predicted_intent": intent_label,
            "confidence": round(float(confidence), 4),
            "predicted_status": predicted_status,
            "lookahead_sec": float(lookahead_sec),
            "safety_mitigation": "DECEL_MAINTAIN_DISTANCE" if intent_label == "INTENT_HARD_BRAKE" else "YIELD_LANE",
            "safety_command": status_command,
        }
        self.socket.sendto(json.dumps(payload).encode("utf-8"), self.address)
        return payload

    def close(self) -> None:
        self.socket.close()
