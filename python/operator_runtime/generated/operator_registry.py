# Placeholder overwritten by tools/generate_operator_artifacts.py during CMake configure.
OPERATORS = []


def get_operator(name: str):
    for op in OPERATORS:
        if op["name"] == name:
            return op
    raise KeyError(name)

