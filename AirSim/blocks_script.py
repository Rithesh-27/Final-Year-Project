import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

BLOCKS_PATH = BASE_DIR / "Blocks" / "WindowsNoEditor" / "Blocks.exe"

def start_blcoks():
    subprocess.Popen([BLOCKS_PATH])
    print("Blocks.exe started\n\n")
    time.sleep(5)

def stop_blocks():
    subprocess.run(
        ["taskkill", "/F","/IM", "Blocks.exe"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    print("Blocks.exe stopped\n\n")
