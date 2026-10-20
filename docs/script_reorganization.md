# Historical script reorganization proposal

The original proposal is preserved in
[archive/design/script_reorganization.md](archive/design/script_reorganization.md).
Its target CLI and default plotting behavior were not fully implemented.

The current [main entry](../scripts/main.py) uses positional repeat/output/worker
arguments, --base-seed with default None, and writes scientific CSV/JSON.
It does not expose --seed, --plots/--no-plots, or runtime torque selectors.

Use [engineering_contract.md](engineering_contract.md) for the current source
owners, artifact paths and diagnostic commands. This page is a historical
pointer, not an outstanding instruction to change scientific behavior.
