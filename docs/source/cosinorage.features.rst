cosinorage.features Module
==========================

Module Contents
---------------

.. automodule:: cosinorage.features
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

.. autoclass:: cosinorage.features.features.WearableFeatures
   :members:

.. autoclass:: cosinorage.features.bulk_features.BulkWearableFeatures
   :members:

Utility Functions
-----------------

Cosinor (Circadian Rhythm Analysis) Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.features.utils.cosinor_analysis.cosinor_multiday

.. autofunction:: cosinorage.features.utils.cosinor_analysis.cosinor_model

.. autofunction:: cosinorage.features.utils.cosinor_analysis.fit_cosinor

Non-parametric (Circadian Rhythm Analysis) Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.features.utils.nonparam_analysis.IV

.. autofunction:: cosinorage.features.utils.nonparam_analysis.IS

.. autofunction:: cosinorage.features.utils.nonparam_analysis.RA

.. autofunction:: cosinorage.features.utils.nonparam_analysis.M10

.. autofunction:: cosinorage.features.utils.nonparam_analysis.L5

Physical Activity Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.features.utils.physical_activity_metrics.activity_metrics

Sleep Metrics
~~~~~~~~~~~~~

.. autofunction:: cosinorage.features.utils.sleep_metrics.apply_sleep_wake_predictions

.. autofunction:: cosinorage.features.utils.sleep_metrics.WASO

.. autofunction:: cosinorage.features.utils.sleep_metrics.TST

.. autofunction:: cosinorage.features.utils.sleep_metrics.PTA

.. autofunction:: cosinorage.features.utils.sleep_metrics.NWB

.. autofunction:: cosinorage.features.utils.sleep_metrics.SOL

.. autofunction:: cosinorage.features.utils.sleep_metrics.SRI

Rescaling Functions
~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.features.utils.rescaling.min_max_scaling_exclude_outliers

Visualization Functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.features.utils.visualization.plot_sleep_predictions

.. autofunction:: cosinorage.features.utils.visualization.plot_non_wear

.. autofunction:: cosinorage.features.utils.visualization.plot_cosinor

Dashboard Functions
~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.features.utils.dashboard.dashboard

