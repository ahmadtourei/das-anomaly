"""
Centralised settings object.  Import this *everywhere* instead of
importing config_defaults or user_config directly.
"""
from importlib import import_module
from types import SimpleNamespace

# 1. start with shipped defaults
from . import config_defaults as _defaults
_settings = {k: v for k, v in vars(_defaults).items() if k.isupper()}

# 2. try to pull in user overrides (optional)
try:
    user_cfg = import_module("das_anomaly.config_user")  # module must be on PYTHONPATH
    _settings.update({k: v for k, v in vars(user_cfg).items() if k.isupper()})
except ModuleNotFoundError:
    pass  # user chose to keep defaults

# 3. freeze into an immutable, dot-accessible object
SETTINGS = SimpleNamespace(**_settings)
del _settings
