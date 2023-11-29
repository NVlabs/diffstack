"""Basic config file"""
import yaml
from easydict import EasyDict
from typing import Dict

class Config:
    """Basic config class"""

    def __init__(self, cfg_path):
        """Load config file"""

        with open(cfg_path, "r", encoding="utf-8") as file_handle:
            self.yml_dict = EasyDict(yaml.safe_load(file_handle))

        # format the config for print
        with open(cfg_path, "r", encoding="utf-8") as file_handle:
            self.format_str = file_handle.read().splitlines()

    def __getattribute__(self, name):
        """Retrieve a value from the config"""

        yml_dict = super().__getattribute__("yml_dict")
        if name in yml_dict:
            return yml_dict[name]

        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        """Set a value from the config"""

        try:
            yml_dict = super().__getattribute__("yml_dict")
        except AttributeError:
            return super().__setattr__(name, value)

        if name in yml_dict:
            yml_dict[name] = value
            return None

        return super().__setattr__(name, value)

    def get(self, name, default=None):
        """Retrieve a value from the config"""

        if hasattr(self, name):
            return getattr(self, name)

        return default


class DictConfig(Config):

    def __init__(self, cfg_dict: Dict):
        """Load config file"""

        self.yml_dict = EasyDict(cfg_dict)

        # format the config for print
        self.format_str = ""
