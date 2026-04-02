"""Run the Awesome-O premise chat CLI.

From the **repo root**, use Python explicitly so flags are handled by ``argparse``
(``sys.argv``) inside ``awesome_o.cli``::

    python run_premise_agent.py
    python run_premise_agent.py --help
"""

import sys

from awesome_o.cli.premise import main

if __name__ == "__main__":
    main(sys.argv[1:])
