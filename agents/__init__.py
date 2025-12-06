# Import modules to ensure @register_agent decorators run
from . import planner  # noqa: F401
from . import analyst  # noqa: F401
from . import critic   # noqa: F401

# Optional plugins (add more as you create them)
try:
    from . import writer  # noqa: F401
except Exception:
    pass

