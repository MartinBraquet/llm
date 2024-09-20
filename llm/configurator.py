import json
from ast import literal_eval
from dataclasses import dataclass

from llm.utils import DataclassUtils


@dataclass
class FileConfig(DataclassUtils):
    config_file: str = None

    def __post_init__(self):
        if self.config_file is not None:
            config = json.load(open(self.config_file))
            for k, v in config.items():
                if not hasattr(self, k):
                    raise ValueError(f"Unknown config key: {k}")
                try:
                    v = literal_eval(v)
                except (SyntaxError, ValueError):
                    pass
                prev_v = getattr(self, k)
                if type(v) != type(prev_v):
                    print(f"WARNING: Key {k} has type {type(v)} but expected {type(prev_v)}")
                print(f"Overriding: {k} = {v}")
                setattr(self, k, v)

    @classmethod
    def from_config(cls, config):
        relevant_config = get_relevant_config(cls, config)
        return cls(**relevant_config)


def get_relevant_config(subclass, config):
    relevant_keys = subclass.keys()
    all_keys = config.keys()
    keys = set(relevant_keys) & set(all_keys)
    relevant_config = {k: getattr(config, k) for k in keys}
    return relevant_config
