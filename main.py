"""Legacy entry point — delegates to madmario.cli.

Kept for backwards compatibility with existing docs/scripts.
Use `python train.py train` or the `madmario` console script.
"""
from madmario.cli import app

if __name__ == "__main__":
    app(["train"])
