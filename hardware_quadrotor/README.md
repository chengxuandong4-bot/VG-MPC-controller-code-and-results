# Curated quadrotor hardware source

The source package is kept under its original Python import name, `rpi_pixhawk_stage4`. For inspection or offline tests, add this directory to `PYTHONPATH` before importing that package. The nested `rpi_pixhawk_stage4/README.md` describes the original Stage 4 safety modes and operating constraints.

This is a curated source snapshot, not a self-contained flight deployment. In particular, `rpi_pixhawk_stage4/controller/candidate_adapter.py` refers to `sitl_outer_loop_stage/python_controller/controller_interface_defaults.json`, which is not included in this repository. Tests or paths that require that external file cannot be claimed as passing here. The raw flight and motion-capture records are also not included.

No hardware command should be run from this release without platform-specific integration, authorization, and safety review.
