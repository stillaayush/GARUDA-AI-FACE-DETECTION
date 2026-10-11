from pathlib import Path


DIRECTORIES = [
    "models",
    "dataset",
    "pipeline",
    "api",
    "v2x",
    "hardware",
    "data/videos",
    "data/imu",
    "checkpoints",
]


def main() -> None:
    root = Path(__file__).parent
    for directory in DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)
    for package in ("models", "dataset", "pipeline", "api", "v2x"):
        init_file = root / package / "__init__.py"
        init_file.touch(exist_ok=True)
    print(f"GARUDA workspace ready at {root.resolve()}")
    print("Next: python generate_synthetic_data.py && python train.py")


if __name__ == "__main__":
    main()
