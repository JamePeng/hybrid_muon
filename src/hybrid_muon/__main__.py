import argparse

from .inspection import inspect_checkpoint


def main():
    parser = argparse.ArgumentParser(prog="python -m hybrid_muon")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Preview groups from a local safetensors checkpoint")
    inspect.add_argument("directory", help="Directory containing model.safetensors or its shard index")
    inspect.add_argument("--verbose", action="store_true", help="List every tensor name and shape")
    args = parser.parse_args()
    try:
        print(inspect_checkpoint(args.directory, verbose=args.verbose))
    except ModuleNotFoundError as exc:
        if exc.name != "safetensors":
            raise
        parser.exit(1, "Checkpoint inspection requires safetensors. Run: python -m pip install safetensors\n")
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"Cannot inspect checkpoint: {exc}\n")


if __name__ == "__main__":
    main()
