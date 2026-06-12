"""Compatibility shim — the MCP server now lives in madmario.mcp_server.

Keeps existing .claude/settings.json configurations
(`python mcp_server.py`) working unchanged.
"""
from madmario.mcp_server import *  # noqa: F401,F403
from madmario.mcp_server import _main, update_state  # noqa: F401

if __name__ == "__main__":
    import asyncio
    asyncio.run(_main())
