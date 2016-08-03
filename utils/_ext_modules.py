import numpy
import pyximport

pyximport.install(setup_args={
                              "include_dirs": numpy.get_include(),
                              "verbose": False,
                              },
                  reload_support=True)

from pos_to_deg import *

