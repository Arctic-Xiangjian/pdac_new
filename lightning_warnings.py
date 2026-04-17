import warnings


def configure_lightning_warning_filters() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"`ModuleAvailableCache` is a special case of `RequirementCache`\..*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API\..*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Deprecated call to `pkg_resources\.declare_namespace\('.*'\)`\..*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Implicitly cleaning up <TemporaryDirectory '.*'>",
        category=ResourceWarning,
    )
