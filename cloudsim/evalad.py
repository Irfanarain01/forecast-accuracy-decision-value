"""Paths of the trained forecaster zoos (one folder per trace); edit if you move the package."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
DIRS = {t: os.path.join(HERE, "zoo", t) for t in ("Alibaba", "Azure", "fastStorage", "Rnd")}
