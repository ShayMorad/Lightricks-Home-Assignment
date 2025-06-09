"""
config_loader.py – read, validate and surface‑check JSON configuration files for the *edit‑image* CLI.

A **configuration file** is a JSON object that tells the program where to
read an input image from, where to write the output, whether to display a
preview window, and—most importantly—which *operations* to run and with
what parameters.  The exact structure is enforced in two layers:

1. **JSON‑Schema (CONFIG_STRUCTURE)** – covers keys, types, required
   parameters and value ranges.
2. **Lightweight logic** – checks that the input path exists, at least
   one of *output*/**display** is active, and that every operation type
   is from the allowed set.

On success, `load_config(path)` returns the parsed dictionary ready for use by the *Filters* pipeline.
On failure, it raises *meaningful* exceptions so the CLI can report them clearly.
"""

import json
import jsonschema
import os

# ---------------------------------------------------------------------------
# Public symbols to export when the module is imported with "*"
# ---------------------------------------------------------------------------
__all__ = [
    "ALLOWED_OPS",
    "CONFIG_STRUCTURE",
    "load_config",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_OPS = {"brightness", "contrast", "saturation", "box", "sharpen", "sobel"}

# The JSON‑Schema that defines correct structure for the whole config file.
# External tools (VSCode extensions, GitHub actions, etc.) can also read it to
# provide live validation.
CONFIG_STRUCTURE = {
    "type": "object",  # the whole input json cfg file should be of object type (i.e. Python dict)
    "properties": {
        "input": {"type": "string"},  # path to input image
        "output": {"type": "string"},  # path to save output (optional)
        "display": {"type": "boolean"},  # show preview window or not
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "value": {"type": "number"},
                    "width": {"type": "integer", "minimum": 1, "maximum": 31},
                    "height": {"type": "integer", "minimum": 1, "maximum": 31},
                    "alpha": {"type": "number", "minimum": 0.0, "maximum": 5.0}
                    # additional properties are allowed by adding them manually under operations (above this comment) and so they will be checked when the config file is loaded
                },
                "required": ["type"],
                "additionalProperties": False,

                # Per‑operation rules ---------------------------------------------------
                "allOf": [
                    {   # box blur
                        "if": {"properties": {"type": {"const": "box"}}, "required": ["type"]},
                        "then": {"required": ["width", "height"]}
                    },
                    {   # sobel (edge detection) – no params
                        "if": {"properties": {"type": {"const": "sobel"}}, "required": ["type"]},
                        "then": {"required": []}
                    },
                    {   # sharpen – requires alpha
                        "if": {"properties": {"type": {"const": "sharpen"}}, "required": ["type"]},
                        "then": {"required": ["alpha"]}
                    },
                    {   # value‑based ops (brightness / contrast / saturation)
                        "if": {"properties": {"type": {"const": "brightness"}}, "required": ["type"]},
                        # brightness should have the parameter value
                        "then": {"required": ["value"]}
                    },
                    {
                        "if": {"properties": {"type": {"const": "contrast"}}, "required": ["type"]},
                        # contrast should have the parameter value
                        "then": {"required": ["value"]}
                    },
                    {
                        "if": {"properties": {"type": {"const": "saturation"}}, "required": ["type"]},
                        # saturation should have the parameter value
                        "then": {"required": ["value"]}
                    }
                ]
            }
        }
    },
    "required": ["input", "output", "display", "operations"],
    "additionalProperties": False
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    """Load *and* validate a configuration JSON file.

    Parameters
    ----------
    path : str | Path
        File system path to the JSON configuration file.

    Returns
    -------
    dict
        The parsed configuration, guaranteed to match `CONFIG_STRUCTURE`
        and the extra logical checks performed in :pyfunc:`validate_config`.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the JSON cannot be parsed or the schema / logical checks fail.
    """

    # 1. Read file -----------------------------------------------------------
    try:
        with open(path, encoding="utf-8") as file:
            cfg_file = json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e.msg}") from None
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file '{path}' not found")

    # 2. Validate structure + logic -----------------------------------------
    validate_config(cfg_file)
    return cfg_file


def validate_config(cfg_file: dict) -> None:
    """Validate *cfg* against the schema and extra run‑time rules.

    This function is intentionally *silent* on success and raises `ValueError`
    or `FileNotFoundError` on failure so that callers can just try/except it.
    """
    # 1. Schema -----------------------------------------------------------
    try:
        jsonschema.validate(instance=cfg_file, schema=CONFIG_STRUCTURE)
    except jsonschema.ValidationError as e:
        raise_errors(e, cfg_file)

    # 2. Logical rules ----------------------------------------------------
    # Input file must exist
    if not os.path.isfile(cfg_file["input"]):
        raise FileNotFoundError(f"Input image '{cfg_file['input']}' does not exist.")

    # At least one of output or display must be active
    if not cfg_file["output"] and not cfg_file["display"]:
        raise ValueError("Config must include either a non-empty 'output' path or 'display' set to true (or both).")

    # Check only supported operations are present
    for i, op in enumerate(cfg_file["operations"]):
        if op["type"] not in ALLOWED_OPS:
            err = "Operation at index " + str(i) + " has unsupported type " + op[
                'type'] + ". Allowed types are: " + ', '.join(sorted(ALLOWED_OPS))
            raise ValueError(err)


# ---------------------------------------------------------------------------
# Helper for nicer schema‑error messages
# ---------------------------------------------------------------------------
def raise_errors(err: jsonschema.ValidationError, cfg_file: dict) -> None:
    """
    Turn a low‑level *jsonschema* ValidationError into a friendly message.

    :param cfg_file: the loaded config file
    :param err: the ValidationError raised by `jsonschema.validate`
    :raises ValueError: always – with a re‑formatted message
    """
    # `err.absolute_path` is a deque like ['operations', 0, 'width']
    path = list(err.absolute_path)

    # Was the failure inside the operations array?
    if "operations" in path:
        # index of the operation that failed
        op_idx = path[path.index("operations") + 1]

        # The actual dict that represents the offending operation
        # (e.g. {"type": "box", "height": 5})
        operations: dict = cfg_file.get("operations")
        op_name = operations[op_idx].get("type", f"operation at index {op_idx}").capitalize()

        # ------------------------------------------------------------------ #
        # 1. Missing parameter (validator == "required")
        #    → "'width' is a required property"
        # ------------------------------------------------------------------ #
        if err.validator == "required":
            missing_param = err.message.split("'")[1]
            err = op_name + " operation (index " + str(op_idx) + ") requires the parameter " + missing_param + "."
            raise ValueError(err) from None

        # ------------------------------------------------------------------ #
        # 2. Wrong type (validator == "type")
        #    → "'hello' is not of type 'number'"
        # ------------------------------------------------------------------ #
        if err.validator == "type":
            bad_field = path[-1]
            expected = err.validator_value  # e.g. 'number'
            err = op_name + " operation (index " + str(
                op_idx) + ") - parameter '" + bad_field + "' must be of type " + expected + "."
            raise ValueError(err) from None

    #     # otherwise it was in the first part regarding input, output and display parameters
    # elif ......:
    #     # write code for handling other cases

    # ---------------------------------------------------------------------- #
    # Fallback – we do not recognise the pattern, so just bubble the message
    # ---------------------------------------------------------------------- #
    err = "Schema validation error: " + err.message + "."
    raise ValueError(err) from None

# # small test
# import tempfile
#
# if __name__ == "__main__":
#     # test_valid_config()
#     # test_invalid_missing_required_field()
#     config = {
#         "input": "image.jpg",  # missing on purpose
#         "output": "result.jpg",
#         "display": True,
#         "operations": [{"type": "sobel"},{"type": "sobel"},{"type": "sobel"},{"type": "ssssobel"}, {"type": "box", "width": 1, "height": 1}]
#     }
#     with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
#         json.dump(config, tmp)
#         tmp_path = tmp.name
#     load_config(tmp_path)
