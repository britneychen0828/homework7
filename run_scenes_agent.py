"""Run the Awesome-O scenes batch CLI.

From the **repo root**::

    python run_scenes_agent.py --project projects\\<id>
    python run_scenes_agent.py --help
"""

import sys

from awesome_o.cli.scenes import main

if __name__ == "__main__":
    main(sys.argv[1:])
