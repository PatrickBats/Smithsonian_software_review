"""Public data configurations; recorded artifact IDs remain unchanged externally."""

SEED = 20261013
VARIANTS = {
    "A1": {"experiment": "m1_positive_grid", "name": "Positive ROI grid tiles"},
    "A2": {
        "experiment": "m2_retile_matched",
        "name": "Context crops, original-coverage anchors",
    },
    "A3": {
        "experiment": "m2_retile_all",
        "name": "Context crops, all training anchors",
    },
}
CONFIGURATIONS = {
    "baseline": {"name": "All original ROI grid tiles", "training": "historical_grid"},
    **{name: {**item, "training": "fixed_steps"} for name, item in VARIANTS.items()},
}
EXPERIMENTS = tuple(item["experiment"] for item in VARIANTS.values())


def resolve_configuration(value):
    """Normalize a public name without treating the three variants as new models."""
    name = "baseline" if value.lower() == "baseline" else value.upper()
    if name not in CONFIGURATIONS:
        raise ValueError("Choose baseline, A1, A2 or A3.")
    return name


def resolve_variant(value):
    """Resolve A1/A2/A3 or their frozen artifact IDs for existing internal contracts."""
    if value.upper() in VARIANTS:
        return VARIANTS[value.upper()]["experiment"]
    if value in EXPERIMENTS:
        return value
    raise ValueError("Choose Method A data variant A1, A2 or A3.")


def variant_id(experiment):
    return next(
        key for key, value in VARIANTS.items() if value["experiment"] == experiment
    )
