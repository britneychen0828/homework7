"""Run the Awesome-O arc chat CLI (default project: ``projects/premise_20260401_221032``).

From the **repo root**::

    python run_arc_agent.py
    python run_arc_agent.py --project projects\\premise_20260401_221032
    python run_arc_agent.py --help

Arguments are parsed with ``argparse`` (``sys.argv``) in ``awesome_o.cli.arc``.
"""

import sys

from awesome_o.cli.arc import main

if __name__ == "__main__":
    main(sys.argv[1:])
