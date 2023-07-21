from __future__ import absolute_import, print_function

import warnings
def format_warning(message, category, filename, lineno, line=''):
    import pathlib
    return f"{pathlib.Path(filename).name} ({lineno}): {message}\n"
warnings.formatwarning = format_warning
del warnings

from .version import __version__

# TODO: which functions to expose here? all?
from .nms import non_maximum_suppression
from .utils import edt_prob, fill_label_holes, sample_points, calculate_extents, export_imagej_rois, gputools_available
from .geometry import star_dist,   polygons_to_label,   relabel_image_stardist, ray_angles, dist_to_coord
from .sample_patches import sample_patches
from .bioimageio_utils import export_bioimageio, import_bioimageio

def _py_deprecation(ver_python=(3,6), ver_stardist='0.9.0'):
     import sys
     from distutils.version import LooseVersion
     if sys.version_info[:2] == ver_python and LooseVersion(__version__) < LooseVersion(ver_stardist):
         print(f"You are using Python {ver_python[0]}.{ver_python[1]}, which will no longer be supported in StarDist {ver_stardist}.\n"
               f"→ Please upgrade to Python {ver_python[0]}.{ver_python[1]+1} or later.", file=sys.stderr, flush=True)
_py_deprecation()
del _py_deprecation
