"""Run dialogue repair on ``scenes.json`` (deterministic; optional ``--llm``).

From the **repo root**::

    python run_fix_scene_dialogue.py --project projects\\<id>
    python run_fix_scene_dialogue.py --help
"""

import sys

from awesome_o.cli.fix_scene_dialogue import main

if __name__ == "__main__":
    main(sys.argv[1:])
