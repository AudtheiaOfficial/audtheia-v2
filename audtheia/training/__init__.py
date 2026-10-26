"""Training support for the Train tab.

Path: audtheia/training/__init__.py

Everything in this package imports cleanly on a computer that has none of the
optional training libraries. The heavy imports, such as the deep-learning
frameworks, happen inside functions in the engine adapters only, so the desktop
application and the field station never load them unless a training run is
started. Nothing here is imported by the capture pipeline or by any field station
entry point.
"""
