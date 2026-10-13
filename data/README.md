# Local data instructions

The repository contains no raw dataset. The synthetic demo needs no downloads.

For the legacy EIA input route, create `data/raw/` and provide the locally held files:

```text
EIA2021_Jan_Jun.csv
EIA2021_Jul_Dec.csv
EIA2022_Jan_Jun.csv
EIA2022_Jul_Dec.csv
EIA2023_Jan_Jun.csv
EIA2023_Jul_Dec.csv
```

Required columns are `Balancing Authority`, `Date`, `Hour` and `DemandMW`. Dates must use MM/DD/YYYY and hour labels must be integers from 1 to 25. All matching `EIA*.csv` files in the supplied directory are read, so do not place overlapping copies there. Duplicated authority/date/hour keys are an error.

The adapter accepts only complete 24-hour days. It retains gaps on excluded dates and never fabricates timestamps from filenames or compresses missing hours. See the root README for the local-time convention and DST limitations. Missing or implausible demand invalidates affected windows; it is not repaired using future values.

Source: [U.S. EIA Hourly Electric Grid Monitor](https://www.eia.gov/electricity/gridmonitor/). [EIA-930 instructions](https://www.eia.gov/survey/form/eia_930/instructions.pdf) describe hourly reporting conventions. Current downloads may differ from these legacy exports. Obtain the data from its source or the project authors and respect the applicable terms; this repository does not promise that a current download is byte-identical to the supplied historical files.

The original package also contains `WTH_2021.csv`, `WTH_2022.csv`, `WTH_2023.csv` and `IPMAN.csv`. They are deliberately outside this software edition. Adding weather requires a verified station/location, timezone and interval convention. Adding economic indicators requires release-date/vintage information to establish what was available at each forecast origin. They must not be merged using month or hour labels alone.

The software writes input hashes into each local run's metadata so a researcher can identify which files were actually used. Generated predictions can contain real observations and must stay out of public commits unless separately reviewed.
