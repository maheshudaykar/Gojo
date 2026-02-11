"""Entry point for the Gojo CLI."""
from __future__ import annotations

from phish_detector.cli import main as cli_main


def main() -> int:
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
