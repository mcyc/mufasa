"""
This module hosts _molecular_constants used to build specific molecular line model classes in
:mod:`SpectralModels <mufasa.spec_models.SpecModels>` using the
:class:`BaseModel <mufasa.spec_models.BaseModel>` and
:class:`BaseModel <mufasa.spec_models.BaseModel>`

"""

#===========================================================================================================
# NH3-specific constants
from pyspeckit.spectrum.models.ammonia import (freq_dict, voff_lines_dict, tau_wts_dict)  # line_names

nh3_constants = {
    'freq_dict': freq_dict,
    'voff_lines_dict': voff_lines_dict,
    'tau_wts_dict': tau_wts_dict
}

#===========================================================================================================
# N2H+ constants taken from pyspeckit/spectrum/models/n2hp.py
# Formatted to match pyspeckit/spectrum/models/ammonia_constants.py
n2hp_constants = dict(
    line_names = ['onezero','twoone','threetwo'],
    freq_dict = {
        'onezero':  93173.7637e6,
        'twoone': 186344.8420e6,
        'threetwo': 279511.8325e6,
    },
    voff_lines_dict = {
        'onezero': [-7.9930, -7.9930, -7.9930, -0.6112, -0.6112, -0.6112, 0.0000, 0.9533,
                    0.9533, 5.5371, 5.5371, 5.5371, 5.9704, 5.9704, 6.9238],
        'twoone': [-4.6258, -4.5741, -4.4376, -4.2209, -4.0976, 3.8808, -3.1619, -2.9453,
                   -2.3469, -1.9290, -1.5888, -1.5516, -1.4523, -1.1465, -0.8065, -0.6532,
                   0.4694, 0.1767, 0.0000, 0.0071, 0.1137, 0.1291, 0.1617, 0.2239, 0.5237,
                   0.6384, 0.7405, 2.1394, 2.5158, 2.5444, 2.6225, 2.8844, 3.0325, 3.0990,
                   3.2981, 3.5091, 3.8148, 3.8201, 6.9891, 7.5057],
        'threetwo': [-3.0666, -2.9296,- 2.7221, -2.6563, -2.5270, -2.4010, -2.2535,
                     -2.1825, -2.1277, -1.5862, -1.0158, -0.6131, -0.6093, -0.5902,
                     -0.4872, -0.4725, -0.2757, -0.0697, -0.0616, -0.0022, 0.0000,
                     0.0143, 0.0542, 0.0561, 0.0575, 0.0687,0.1887, 0.2411, 0.3781,
                     0.4620, 0.4798, 0.5110, 0.5540, 0.7808, 0.9066, 1.6382, 1.6980,
                     2.1025, 2.1236, 2.1815, 2.5281, 2.6458, 2.8052, 3.0320, 3.4963]
    },
    tau_wts_dict = {
        'onezero': [0.025957, 0.065372, 0.019779, 0.004376, 0.034890, 0.071844, 0.259259,
                   0.156480, 0.028705, 0.041361, 0.013309, 0.056442, 0.156482, 0.028705,
                   0.037038],
        'twoone': [0.008272, 0.005898, 0.031247, 0.013863, 0.013357, 0.010419, 0.000218,
                   0.000682, 0.000152, 0.001229, 0.000950, 0.000875, 0.002527, 0.000365,
                   0.000164, 0.021264, 0.031139, 0.000576, 0.200000, 0.001013, 0.111589,
                   0.088126, 0.142604, 0.011520, 0.027608, 0.012800, 0.066354, 0.013075,
                   0.003198, 0.061880, 0.004914, 0.035879, 0.011026, 0.039052, 0.019767,
                   0.004305, 0.001814, 0.000245, 0.000029, 0.000004],
        'threetwo': [0.001845, 0.001818, 0.003539, 0.014062, 0.011432, 0.000089, 0.002204,
                     0.002161, 0.000061, 0.000059, 0.000212, 0.000255, 0.000247, 0.000436,
                     0.010208, 0.000073, 0.007447, 0.000000, 0.000155, 0.000274, 0.174603,
                     0.018683, 0.135607, 0.100527, 0.124866, 0.060966, 0.088480, 0.001083,
                     0.094510, 0.014029, 0.007191, 0.022222, 0.047915, 0.015398, 0.000071,
                     0.000794, 0.001372, 0.007107, 0.016618, 0.009776, 0.000997, 0.000487,
                     0.000069, 0.000039, 0.000010]
    }
)

#===========================================================================================================


