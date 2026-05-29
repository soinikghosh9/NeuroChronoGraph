"""V3 package — kept for the FeatureStreamFusion module.

NOTE: NeuroChronoGraphV3 is not loaded eagerly here because it has a stale
import from V2 (``BandSpecificEncoder`` was removed) and would break the
import chain. The hierarchical V2 model is the production architecture;
only :mod:`src.models.v3.feature_stream_fusion` is reused.
"""
