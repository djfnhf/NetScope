from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG = {
     "IN_ROOT_BASE": [
        str(_PROJECT_ROOT / "datasets" / "ISCX-VPN_app"),
        str(_PROJECT_ROOT / "datasets" / "ISCX-VPN_service")
    ],

    "SPLITCAP_EXE": str(_PROJECT_ROOT / "benchmark" / "SplitCap.exe"),

    "WORKERS": 12,
    "FLOW_SPLIT_WORKERS": 12,
    "PACKET_SPLIT_WORKERS": 12,

    "FLOW_MIN_BYTES": 2048,
    "FLOW_MIN_PKTS": 3,
    "FLOW_CLASS_MIN": 10,
    "FLOW_CLASS_CAP": 500,
    "FLOW_SPLIT_RATIOS": (0.8, 0.1, 0.1),

    "TCP_MIN_L3": 144,
    "UDP_MIN_L3": 103,
    "PKT_CLASS_MIN": 100,
    "PKT_CLASS_CAP": 5000,
    "PKT_SPLIT_RATIOS": (0.8, 0.1, 0.1),

    "SEED": 42,
    "DO_FLOW_STAGE": True,
    "DO_PACKET_FROM_FLOW": True,
}
