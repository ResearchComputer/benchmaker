import asyncio
import os
import random
import sys
from pathlib import Path

from benchmaker import BenchConfig,BenchRunner,ConstantRPS,SandboxWorkloadType,StaticWorkload

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES_FILE = ROOT / ".local" / "images.txt"

def load_images(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(
            f"image list not found: {path}\n"
            f"populate it with: python scripts/pull_swe_images.py --out {path}"
        )
    images = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not images:
        raise SystemExit(f"image list is empty: {path}")
    return images


def pick_image() -> str:
    explicit = os.environ.get("IMAGE")
    if explicit:
        return explicit
    images_file = Path(os.environ.get("IMAGES_FILE", DEFAULT_IMAGES_FILE))
    return random.choice(load_images(images_file))

os.environ.setdefault("RPS", "5")
os.environ.setdefault("DURATION", "30")
os.environ.setdefault("BASE_URL", "https://sandbox.swissai.cscs.ch")

image = pick_image()
print(f"benchmark image: {image}", file=sys.stderr)

r=BenchRunner(
    BenchConfig(
        workload_type=SandboxWorkloadType(
            base_url=os.environ["BASE_URL"],
            spec={"image": image},
            ttl_seconds=600
        ),
        workload=StaticWorkload(
            items=["echo hello","uname -a",["python","-c","print(2+2)"]]
        ),
        load=ConstantRPS(
            rps=float(os.environ["RPS"]),
            duration_s=float(os.environ["DURATION"]))
        )
)

asyncio.run(r.run())

r.metrics.render(sys.stdout)