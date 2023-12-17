import importlib


class Struct(dict):
    def __getattr__(self, key):
        try:
            value = self[key]
            if type(value) == type({}):
                return Struct(value)
            return value

        except KeyError:
            raise AttributeError(key)


def set_cfg_from_file(cfg_path='configs/config_case_luding.py'): 
    module_spec = importlib.util.spec_from_file_location('cfg_file', cfg_path) 
    module = importlib.util.module_from_spec(module_spec) 
    module_spec.loader.exec_module(module)    
    cfg = module.cfg
    cfg = Struct(cfg)

    return cfg

cfg = set_cfg_from_file()
