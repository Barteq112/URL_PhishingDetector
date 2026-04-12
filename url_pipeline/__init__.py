__all__ = ["extract_features"]


def __getattr__(name: str):
    if name == "extract_features":
        from url_pipeline.extractor import extract_features

        return extract_features
    raise AttributeError(f"module 'url_pipeline' has no attribute '{name}'")
