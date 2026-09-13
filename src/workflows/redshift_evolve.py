"""Deprecated import location for the shared flat-LambdaCDM cosmology."""

from .cosmology import FlatLambdaCDM

RedshiftEvolution = FlatLambdaCDM

__all__ = ["FlatLambdaCDM", "RedshiftEvolution"]
