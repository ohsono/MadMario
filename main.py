"""Legacy entry point — delegates to train.py.

Kept for backwards compatibility with existing docs/scripts.
Use `python train.py train` for the full-featured CLI.
"""
from train import app

if __name__ == "__main__":
    app(["train"])
