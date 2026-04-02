"""Run the scenes rewrite pass → ``scenes_rewrite.json``.

From the **repo root**::

    python run_scenes_rewrite.py --project projects\\<id>
    python run_scenes_rewrite.py --help
"""

import sys

from awesome_o.cli.scenes_rewrite import main

if __name__ == "__main__":
    main(sys.argv[1:])
