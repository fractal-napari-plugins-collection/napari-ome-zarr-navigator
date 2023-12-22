from functools import lru_cache

import zarr


@lru_cache(maxsize=16)
def get_metadata(zarr_url):
    with zarr.open(zarr_url) as zarr_attrs:
        return dict(zarr_attrs.attrs)
