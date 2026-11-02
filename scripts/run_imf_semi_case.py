"""Single-halo CLI adapter for the archived comparison-only migration model."""
from workflows.cli import build_config_from_arguments, build_parser
from workflows.config import SEMI_WITHOUT_TRAP
from workflows.workflow import run_simulation

args=build_parser().parse_args()
run_simulation(build_config_from_arguments(args, disk_model=SEMI_WITHOUT_TRAP))
