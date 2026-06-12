"""MadMario — Double DQN Super Mario Bros agent with multi-agent training.

Public API:
    Config tree        madmario.config.Config
    Environment        madmario.environment.make_env
    Agent              madmario.agent.Mario
    Multi-agent        madmario.multi_agent.run_multi_agent
    CLI                madmario.cli.app   (console script: `madmario`)
"""
__version__ = "2.0.0"

from madmario.config import Config  # noqa: F401
