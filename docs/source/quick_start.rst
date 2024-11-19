Getting Started
===============

This guide helps you get started with **MUFASA** in minutes.

First Steps
-----------
To perform an NH₃ (1,1) fit automatically with up to two components, run the following:

.. code-block:: python

    from mufasa import master_fitter as mf
    reg = mf.Region(cubePath, paraNameRoot, paraDir, fittype='nh3_multi_v')
    reg.master_2comp_fit(snr_min=0)

Explore More
------------
- Learn about MUFASA's capabilities in the :doc:`Tutorials <tutorials/index>`.
- See the full list of supported modules in the :doc:`API Reference <api/modules>`.
