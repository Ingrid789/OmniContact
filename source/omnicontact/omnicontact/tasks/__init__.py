"""Package containing task implementations for various robotic environments."""

# Gym registrations use string entry points. Load configurations only after
# AppLauncher starts Isaac Sim and the user selects a task.
from . import omnicontact  # noqa: F401
