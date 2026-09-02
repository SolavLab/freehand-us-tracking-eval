"""Evaluation of visual-inertial pose-tracking pipelines against a
motion-capture reference, for freehand 3D-ultrasound probe localization.

Nothing in this package imports matplotlib.pyplot. Figures live in
``vipose.report.figures`` and are drawn only from the command line, after
results have been written to disk.
"""

__version__ = "1.0.0"

from .recordings import Cell, Dataset, DatasetError

__all__ = ["Cell", "Dataset", "DatasetError", "__version__"]
