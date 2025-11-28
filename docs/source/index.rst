.. mne_tesa documentation master file

mne_tesa documentation
======================

Welcome to MNE-TESA!

MNE-TESA is an adaptation of the TMS-EEG signal analyzer (TESA) from Rogasch et al., 2017, built as a plugin to MNE-Python.

See https://github.com/AlexanderEngelmark/mne-TESA/blob/main/tutorial.ipynb for a tutorial on how to use the functions.

**Reference Paper:**
N. C. Rogasch, C. Sullivan, R. H. Thomson, N. S. Rose, N. W. Bailey, P. B.
Fitzgerald, F. Farzan, and J. C. Hernandez-Pavon, “Analysing concurrent
transcranial magnetic stimulation and electroencephalographic data: A
review and introduction to the open-source tesa software,” NeuroImage,
vol. 147, 2 2017.

TESA functions adapted in MNE-Python
------------------------------------
.. autosummary::
   :toctree: _autosummary

   mne_tesa.find_pulse
   mne_tesa.notch_filter_epochs
   mne_tesa.tesa_interp_cubic_spline
   mne_tesa.tesa_replace_constant_amplitude
   mne_tesa.mne_tesa_class_comp
   mne_tesa.tesa_ica_select
   mne_tesa.ICA_TESA.find_bads_electrode_noise
   mne_tesa.ICA_TESA.find_bads_lateral_eye_movement
   mne_tesa.ICA_TESA.find_bads_tms_muscle

Plotting functions for GFP and evoked responses
-----------------------------------------------
.. autosummary::
   :toctree: _autosummary

   plotly_evoked.plot_evoked
   plotly_evoked.plot_gfp

External References
-------------------
- `TESA User Manual <https://nigelrogasch.gitbook.io/tesa-user-manual/>`_
- `MNE-Python <https://mne.tools/stable/index.html>`_

Indices and Tables
==================
* :ref:`genindex`
* :ref:`modindex`
