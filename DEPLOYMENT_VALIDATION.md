# Deployment validation gate

The included model metrics are from CSI-Bench H5 files, not the ESP32 stream.
They must not be used as a healthcare performance claim.  The runtime now
blocks voice alerts when recent ESP32 features fall outside the training
distribution, rather than treating extrapolated model scores as confidence.

Before enabling alerts, collect a new, labelled ESP32 dataset using the exact
firmware, Wi-Fi channel, receiver, placement, and 116-amplitude/116-phase
output contract used in deployment.  Record packet loss, timestamp, subject,
room, activity, and a synchronized ground-truth fall label.

Use subject-disjoint train/validation/test splits.  The held-out test set must
include quiet rooms, walking, sitting, lying down, transfers, dropped objects,
visitors, pets, radio interference, and multiple fall directions.  Calibrate
normalization and all thresholds only on the training/validation sets; lock
them before the final test.

For the requested 98% target, predefine separate requirements for fall recall
and false-alert rate, with confidence intervals, rather than relying on a
single accuracy number.  Review every missed fall and every false alert.  Keep
the confirmation/alert path supervised during a pilot, and do not enable
autonomous clinical use until the locked, device-specific test meets the
approved safety criteria.
