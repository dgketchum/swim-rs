import numpy as np


def kcb_daily(tracker, day_data):
    """Compute basal crop coefficient (Kcb) from NDVI.

    Currently uses a sigmoid relationship bounded by `kc_max`, with slope `ndvi_k`
    and inflection `ndvi_0`. Alternative linear mapping is noted in comments.

    Parameters
    - tracker: SampleTracker (provides ndvi_k, ndvi_0, kc_max).
    - day_data: DayData with `ndvi` 1xN array for the day.
    """

    # linear
    # tracker.kc_bas = day_data.ndvi * tracker.ndvi_beta + tracker.ndvi_alpha
    # sigmoid
    tracker.kc_bas = tracker.kc_max / (
        1 + np.exp(-tracker.ndvi_k * (day_data.ndvi - tracker.ndvi_0))
    )
    tracker.kc_bas = np.minimum(tracker.kc_max, tracker.kc_bas)
