from typing import Any, Callable, Dict

# begin-env-vars-definition

environment_variables: Dict[str, Callable[[], Any]] = {
    # TODO: Add environment variables here
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
