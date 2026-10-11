# PROJECT GARUDA Setup Guide

## 1. Open the project

Open this folder in Visual Studio Code and open the integrated terminal with `Ctrl + ~` on Windows/Linux or `Cmd + ~` on macOS.

## 2. Generate the workspace

The repository already includes the scaffold. To recreate missing folders and package markers, run:

```bash
python setup_project.py
```

## 3. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Generate synchronized training data

```bash
python generate_synthetic_data.py
```

This creates `data/videos/`, `data/imu/`, and `data/metadata.csv`.

## 6. Train the model

```bash
python train.py --epochs 5
```

The best checkpoint is saved to `checkpoints/best_intent_model.pth`.

## 7. Run realtime inference

```bash
python main.py
```

Use `python main.py --no-broadcast` to test without UDP V2X messages. Press `Q` to stop the camera loop.
