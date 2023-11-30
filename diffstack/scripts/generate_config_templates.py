"""
Helpful script to generate example config files for each algorithm. These should be re-generated
when new config options are added, or when default settings in the config classes are modified.
"""
import os

import diffstack
from diffstack.configs.registry import EXP_CONFIG_REGISTRY


def main():
    # store template config jsons in this directory
    target_dir = os.path.join(diffstack.__path__[0], "../config/templates/")
    for name, cfg in EXP_CONFIG_REGISTRY.items():
        cfg.dump(filename=os.path.join(target_dir, name + ".json"))


if __name__ == "__main__":
    main()
