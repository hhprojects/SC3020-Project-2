"""project.py - Entry point. Run with:  python project.py

Loads the configuration and opens the window.
"""

from interface import launch_app
from preprocessing import load_config


def main():
    """Load config.json and .env, then open the main window."""
    config = load_config()
    launch_app(config)


if __name__ == "__main__":
    main()
