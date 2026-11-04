"""JSON-backed command options with explicit path origins and run provenance."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

PATH_KEYS = {
    "detector_checkpoint",
    "classifier_checkpoint",
    "dataset",
    "init",
    "out",
    "detector",
    "run",
    "frozen",
    "references",
    "dino_run",
    "spring_source",
    "data",
    "assignments",
    "categories",
    "output",
    "checkpoint",
    "image",
    "boxes",
    "dino_source",
    "specimen_split",
    "broad_map",
    "baseline",
    "crop_root",
    "root",
    "artifact_root",
    "crops",
    "destination",
    "workspace",
    "fine_to_broad",
    "swin_checkpoint",
    "dino_checkpoint",
    "detector_pretrained",
    "derived_data",
    "pipeline_root",
    "specimen",
    "request",
    "resume",
}


def load_config(path):
    """Load JSON settings and resolve path fields against the file's directory.

    Args:
        path: Existing JSON object file.

    Returns:
        Dictionary with absolute path strings and otherwise unchanged values.

    Raises:
        ValueError: JSON does not contain an object.
        FileNotFoundError: Configuration file is missing.
    """
    path = Path(path).expanduser().resolve(strict=True)
    settings = json.loads(path.read_text())
    if not isinstance(settings, dict):
        raise ValueError("Configuration must be a JSON object")
    for key, value in settings.items():
        if key.replace("-", "_") in PATH_KEYS and value is not None:
            candidate = Path(value).expanduser()
            settings[key] = str((path.parent / candidate).resolve())
    return settings


class ConfigArgumentParser(argparse.ArgumentParser):
    """Accept JSON defaults while retaining argparse's types and CLI precedence."""

    def parse_args(self, args=None, namespace=None):
        """Parse command options, validating config keys through registered actions.

        Args:
            args: Optional argument list; defaults to the process command line.
            namespace: Optional argparse namespace to populate.

        Returns:
            Parsed namespace with paths made absolute.

        Raises:
            SystemExit: Unknown settings, invalid types or missing required options.
        """
        argv = list(sys.argv[1:] if args is None else args)
        if "--config" not in self._option_string_actions:
            self.add_argument(
                "--config", type=Path, help="JSON defaults; CLI options take precedence"
            )
        probe = argparse.ArgumentParser(add_help=False)
        probe.add_argument("--config", type=Path)
        config_arg, _ = probe.parse_known_args(argv)
        if config_arg.config:
            values = load_config(config_arg.config)
            actions = {action.dest: action for action in self._actions}
            prefix = []
            for key, value in values.items():
                key = key.replace("-", "_")
                if key not in actions or key in ("config", "help"):
                    self.error(f"Unknown configuration option: {key}")
                action = actions[key]
                if value is None:
                    continue
                if not action.option_strings:
                    self.error(f"Pass positional {key} on the command line")
                flag = action.option_strings[0]
                if any(
                    token == option or token.startswith(option + "=")
                    for token in argv
                    for option in action.option_strings
                ):
                    continue
                if isinstance(action, argparse._StoreTrueAction):
                    if not isinstance(value, bool):
                        self.error(f"{key} must be boolean")
                    if value:
                        prefix.append(flag)
                else:
                    prefix.extend(
                        [flag, *map(str, value)] if isinstance(value, list) else [flag, str(value)]
                    )
            argv = prefix + argv
        result = super().parse_args(argv, namespace)
        for key, value in vars(result).items():
            if key in PATH_KEYS and value is not None:
                resolved = Path(value).expanduser().resolve()
                setattr(result, key, resolved if isinstance(value, Path) else str(resolved))
        return result


def save_effective_config(args, destination):
    """Write resolved settings and file hashes beside newly generated outputs.

    Args:
        args: Parsed command namespace or dictionary.
        destination: New metadata file. Existing different metadata is rejected.

    Returns:
        Path of the immutable metadata record.

    Raises:
        FileExistsError: The destination contains an incompatible record.
    """
    values = vars(args) if isinstance(args, argparse.Namespace) else args
    values = {
        key: str(value) if isinstance(value, Path) else value for key, value in values.items()
    }
    identities = {}
    for key, value in values.items():
        if key in PATH_KEYS and value and Path(value).is_file():
            digest = hashlib.sha256()
            with Path(value).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            identities[key] = {"path": str(Path(value).resolve()), "sha256": digest.hexdigest()}
    payload = (
        json.dumps(
            {
                "schema_version": 2,
                "settings": values,
                "inputs": identities,
                "package_sources": package_fingerprints(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    destination = Path(destination)
    if destination.exists():
        if destination.read_text() != payload:
            raise FileExistsError(f"Incompatible run record: {destination}; choose a new output")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write(payload)
    return destination


def package_fingerprints():
    """Return source hashes for every installed project module.

    Returns:
        Mapping from package-relative Python filename to SHA-256 identity.
        Reading source files is explicit and writes no output.
    """
    package = Path(__file__).resolve().parents[1]
    return {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
