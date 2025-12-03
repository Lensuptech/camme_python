def generate_filter_name(settings: dict) -> str:
    """
    Generate a readable filter parameter list + a unique filter name.

    - Keeps only non-zero numeric values
    - Ignores invalid / non-numeric values
    - Produces a readable string and a compact unique name
    """

    active_params = {}

    for key, value in settings.items():
        # Attempt numeric conversion
        try:
            numeric_val = float(value)

            # Skip zero / no change
            if numeric_val != 0:
                active_params[key] = numeric_val

        except (ValueError, TypeError):
            # Ignore invalid input (strings, None, dicts, etc.)
            continue

    # No active adjustments
    if not active_params:
        return "No_Adjustments"

    # Sort keys for stable output
    sorted_items = sorted(active_params.items())

    # Readable parameter list
    param_list = " | ".join(f"{k}={v:.1f}" for k, v in sorted_items)

    # Unique compact filter name
    unique_name = "_".join(f"{k}{int(round(v))}" for k, v in sorted_items)

    # Limit name length (safe for filesystem)
    if len(unique_name) > 100:
        suffix = abs(hash(unique_name)) % 10000
        unique_name = f"{unique_name[:100]}_h{suffix}"

    return f"{param_list} || FILTER: {unique_name}"