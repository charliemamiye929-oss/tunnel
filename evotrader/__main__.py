"""Allow `python3 -m evotrader ...` as well as the `evotrader` console script."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
