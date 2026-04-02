"""Run the Awesome-O sequence batch CLI.

From the **repo root**::

    python run_sequence_agent.py --project projects\\<id>
    python run_sequence_agent.py --help
"""

import sys

from awesome_o.cli.sequence import main

if __name__ == "__main__":
    main(sys.argv[1:])
