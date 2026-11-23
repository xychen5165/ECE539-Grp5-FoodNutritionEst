# Frozen annotation-use assignments

These are reviewed manifest snapshots, not a replacement for the team's shared
split builder. Source split/grouping commit:
`aec9a6c0f6ef6826185529602a7c6af72b638a80`.

- `train_development.csv`: 454 annotated dishes in team train, 357 plate groups.
- `dev_tuning.csv`: 103 annotated dishes in team dev, 81 groups.
- `assignment.csv`: all 1,056 annotated dish IDs and their permitted roles.
- `protocol.json`: integrity hashes, source commits and exclusion checks.

The remaining annotations comprise 486 test dishes (one previously displayed
example is flagged separately) and 13 non-test dishes linked to test plates.
They are never downloaded by the preparation command or used for tuning.
Recorded access flags describe the protocol's original construction snapshot;
training/dev images have since been used. They are not pristine holdouts.

`data_utils.py` checks manifest fingerprints, unique IDs and group isolation.
The raw annotation index is downloaded separately into ignored `_local/` and
verified against its pinned SHA256. Raw RGBs, masks and nutrient tables are not
included here. Group IDs are scoped to this source commit and are inferred.
