import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import numpy
    import pyximport;


    pyximport.install(setup_args={#"script_args": [  "--quiet",],
                              "include_dirs": numpy.get_include()},
                    reload_support=True)

    from pos_to_deg import *

