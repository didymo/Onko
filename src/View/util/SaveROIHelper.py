def generate_non_duplicated_name(original_name, names):
    """
    This function generates suffix to avoid duplicated names
    Parameters
    ----------
    original_name: the original name
    names: the list of names that original names cannot be duplicated with

    Returns: original name with a suffix
    -------

    """
    changed_name = original_name
    suffix_arr = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    suffix_index = 0
    suffix = ""
    while changed_name in names:
        suffix = suffix + suffix_arr[suffix_index]
        changed_name = original_name + "_" + suffix
        suffix_index = (suffix_index + 1) % len(suffix_arr)
        if suffix_index > 0:
            suffix = suffix[:-1]
    return changed_name
