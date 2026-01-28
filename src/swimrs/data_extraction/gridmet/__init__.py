"""
GridMET meteorological data extraction module.

Provides tools for downloading daily meteorological data from the
GridMET dataset via THREDDS OPeNDAP services.

Key Classes:
    GridMet: Downloads daily meteorological variables for point locations.

Variables Available:
    - etr: ASCE grass reference ET (mm)
    - eto: ASCE alfalfa reference ET (mm)
    - pr: Precipitation (mm)
    - tmmn: Minimum temperature (K)
    - tmmx: Maximum temperature (K)
    - srad: Surface downward shortwave radiation (W/m^2)
    - vs: Wind speed (m/s)
    - sph: Specific humidity (kg/kg)

Example:
    >>> from swimrs.data_extraction.gridmet import GridMet
    >>>
    >>> gm = GridMet(variable="etr", lat=45.5, lon=-116.5,
    ...              start="2020-01-01", end="2020-12-31")
    >>> data = gm.get_point_timeseries()
"""

from swimrs.data_extraction.gridmet.thredds import GridMet

__all__ = ["GridMet"]

if __name__ == "__main__":
    pass
# ========================= EOF ====================================================================
