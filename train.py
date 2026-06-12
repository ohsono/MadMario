#!/usr/bin/env python3
"""Compatibility shim — the CLI now lives in madmario.cli.

    python train.py train ...      (still works)
    madmario train ...             (after `pip install -e .`)
"""
from madmario.cli import app

if __name__ == "__main__":
    app()
