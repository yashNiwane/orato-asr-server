"""Orato ASR Server v2 package.

Sets TF-avoidance env vars before any submodule import: transformers
conditionally imports TensorFlow when it is present in the environment
(e.g. Kaggle images), which is unnecessary for this server and can drag in
broken dependency chains.
"""

import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
