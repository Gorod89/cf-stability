# NOTICE: data sources and the terms that apply to the derived files

The code, the configurations and the documentation of this repository are under the MIT licence (`LICENSE`). The
run outputs published with them (trained checkpoints, calibrations, per-event metrics, audits, corridor scenarios
and corridor results, tables and figures) were computed from third-party data. The MIT licence does not extend to
those files.

This notice reports the terms of the data providers as they were read on 16 November 2026, with the address of
each text. It does not merge the terms of several sources into one licence, and it is not legal advice. A sentence
that is a condition of this project, and not of a provider, is marked as such.

No data set is redistributed here, with one exception: the corridor scenarios hold NGSIM data in adapted form, which
the licence of NGSIM allows (see NGSIM below). No event, trajectory or sample of highD or of the Waymo Open Dataset
is in this repository, in the history of its branches and tags or in its archived releases.

## highD

* Source: Krajewski et al. (2018); levelXdata. Used as the highD car-following events of the FollowNet release
  (Chen et al., 2023). The raw highD recordings were not used.
* Terms of the provider ("Terms and Conditions" of the highD dataset, https://levelxdata.com/highd-dataset/, read
  on 16 November 2026). The dataset is made freely available for non-commercial purposes such as academic research,
  teaching or scientific publications. Point 2: every work that makes use of the dataset includes a reference to
  it. Point 3: the dataset and modified versions are not distributed, and "It is permissible to distribute
  derivative works in as far as they are abstract representations of the highD dataset (such as models trained on it
  or additional annotations that do not directly include any of our data) and do not allow to recover the highD
  dataset or something similar in character." Point 4: neither the dataset nor any derivative work may be used for
  commercial purposes.
* Files with the highD events in their provenance:
  * the checkpoints trained on them, `runs/<experiment>/follownet_highd/**/model.pt`;
  * every checkpoint fine-tuned from such a model: the learned models of the runs on `ngsim_i80` and `ngsim_i80_p0`
    whose `metrics.json` names a highD run in `init_from`;
  * the corridor laws that list such runs as members (`runs/corridor/laws/*.json`) and the corridor results of
    these laws;
  * the per-event metrics (`test_events.parquet`, `transfer_*.parquet`), the audits (`stability.json`,
    `platoon.json`, `certificate.json`, `full_history.json`), `data/calibration/follownet_highd/`, the fold
    definitions `data/splits/follownet_highd_*.json`, and the tables and figures built from them.
* What these files are: model weights, fitted parameters, per-event error metrics and aggregate statistics. None of
  them holds data of the dataset (the checkpoints of the k-nearest-neighbour baseline hold no samples, see below).
* Condition of this project, following point 4 of the provider: every file with the highD events in its provenance
  is provided for non-commercial research and teaching only.

## NGSIM

* Source: Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data, I-80 and US-101 (U.S.
  Department of Transportation, Federal Highway Administration; ITS DataHub, doi:10.21949/1504477). Used as the raw
  trajectories of data.transportation.gov (data set `8ect-6jqj`), reconstructed by this project
  (`docs/decisions.md`, D42-D46), and as the NGSIM I-80 events of the FollowNet release.
* Terms of the provider. The record of the data set on data.transportation.gov names the licence "Creative Commons
  Attribution | Share Alike 3.0 Unported" (licence id `CC_30_BY_SA`, terms link
  http://creativecommons.org/licenses/by-sa/3.0/legalcode; read on 16 November 2026 in the answers of the portal
  for `api/views/8ect-6jqj.json` and in the entry of its `data.json`).
* Files that hold NGSIM data in adapted form: the corridor scenarios `runs/corridor/scenarios/**` (the reconstructed
  trajectories of the study sections `ground_truth.npz`, the vehicle tables `vehicles_truth.npz`, the boundary speed
  series `boundary.npz`, the SUMO demand `routes.rou.xml`, with the network files and the ground-truth statistics of
  the same folders). They are adaptations of the data set and are shared under **CC BY-SA 4.0**
  (https://creativecommons.org/licenses/by-sa/4.0/), a later version of the licence, which Section 4(b) of
  version 3.0 allows for adaptations. Changes made: selection of the study sections and periods, reconstruction and
  smoothing of the trajectories, conversion to SI units, building of the simulation inputs.
* Other files computed from NGSIM: the checkpoints trained on NGSIM events, `data/calibration/ngsim_*/` and
  `data/calibration/follownet_ngsim_i80/`, the law tables `runs/corridor/laws/*.npz` (fitted parameters per
  vehicle), the corridor laws and the corridor results `runs/corridor/**`, the per-event metrics
  (`transfer_ngsim_*.parquet` and those of the runs on these sets) and the fold definitions. They hold no NGSIM
  data and remain subject to the terms of this source; this notice does not state their legal status under it.

## Files computed from highD and NGSIM together

A checkpoint fine-tuned on NGSIM I-80 from a model trained on the highD events was computed from both sources and
remains subject to the terms of each of them. The same holds for the corridor laws built from such checkpoints and
for the results of these laws on the NGSIM scenarios. This notice does not state under which single licence such a
file may be used or passed on: the highD terms allow non-commercial use only, and the NGSIM record carries a
ShareAlike licence. Use for non-commercial research and teaching, with the attributions below, is the use that
both sets of terms allow, and it is the only use for which this project provides these files.

## Waymo Open Dataset

* Source: the car-following pairs of Hu et al. (2021; Mendeley Data, doi:10.17632/wfn2c3437n.2) and the Waymo event
  set of the FollowNet release. Used for evaluation only: the transfer of the models trained on highD (experiment
  E3) and the calibration of the IDM. No neural model of this repository was trained on Waymo data.
* Terms of the provider: the Waymo Dataset License Agreement for Non-Commercial Use (March 2025),
  https://waymo.com/open/terms/, read on 16 November 2026. A copy of the Agreement is in
  `WAYMO_OPEN_DATASET_LICENSE.txt`. The Mendeley record of the pairs carries a CC BY 4.0 label; the Agreement covers
  any data of the data set, and this project applies it to everything computed from the pairs.
* Files: the IDM parameters fitted on the pairs (`data/calibration/waymo/` and `data/calibration/follownet_waymo/`:
  the global fit, the fits per event and their spread), the per-event metrics `transfer_waymo.parquet`, the fold
  definitions `data/splits/*waymo*.json`, and the rows of the tables built from them.
* Notice required by the Agreement. These files are released under the Waymo Dataset License Agreement for
  Non-Commercial Use. Any further downstream use or modification of them is subject to the terms of that Agreement,
  including the non-commercial restrictions of its Section 4. The attribution is given below.

## OpenACC

* Source: Makridis et al. (2021); Joint Research Centre of the European Commission. Used as the car-following
  campaigns (event sets `openacc`, `openacc_acc`, `openacc_human`) and as leader profiles of the platoon test.
* Terms of the provider. The record "Open ACC Database" of the JRC Data Catalogue (doi:10.2905/JRC.KMH3D00, read on
  16 November 2026) gives the use conditions "European Commission reuse notice". The copyright notice distributed
  with the data (`copyright.txt` in every campaign folder of the download location that the record names;
  downloaded on 27 September 2026) licenses the data set under CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/): reuse is allowed provided appropriate credit is given and any
  changes are indicated.
* Files: the checkpoints `runs/e5*/openacc_*/**/model.pt`, the platoon tests `platoon.json` (statistics per
  profile), `data/calibration/openacc*/`, the fold definitions. No series of the data set is in the repository.
  Changes made are described in `docs/data_contract.md`.

## FollowNet

* Source: Chen et al. (2023); figshare collection doi:10.6084/m9.figshare.c.6777810. Used as the released event
  sets of HighD, NGSIM I-80 and Waymo.
* figshare labels the items of the collection CC0 (read on 16 November 2026). The event sets were derived from
  highD, NGSIM and the Waymo Open Dataset, and this project applies the terms of those sources to everything
  computed from the events, as set out in the sections above. No event set is redistributed here.

## What is not in this repository

* The raw data and the extracted events (`data/raw/`, `data/events/` are excluded from git) and the simulated
  trajectories of the corridor runs.
* Samples of the training data inside checkpoints: the checkpoint of the k-nearest-neighbour baseline holds the
  scaler, the number of samples and the seed only (`docs/decisions.md`, D137); the model is re-fitted from the events
  when it is used, which needs the data (level 3 of the README).

## Which file derives from which data

A training run lives in `runs/<experiment>/<data>/<model>/<run>/`: `<data>` names the event set it was trained on,
and `init_from` in its `metrics.json` names the run it was fine-tuned from. The corridor laws
(`runs/corridor/laws/`) list their member runs, and a corridor result lives in
`runs/corridor/<scenario>/<law>/<seed>/`.

## Attributions

* highD: Krajewski, R.; Bock, J.; Kloeker, L.; Eckstein, L. The highD Dataset: A Drone Dataset of Naturalistic Vehicle
  Trajectories on German Highways for Validation of Highly Automated Driving Systems. In Proceedings of the 2018 21st
  International Conference on Intelligent Transportation Systems (ITSC), 2018; pp. 2118-2125.
  doi:10.1109/ITSC.2018.8569552.
* NGSIM: U.S. Department of Transportation Federal Highway Administration. (2016). Next Generation Simulation (NGSIM)
  Vehicle Trajectories and Supporting Data. [Dataset]. Provided by ITS DataHub through Data.transportation.gov.
  Accessed 2026-09-27 from http://doi.org/10.21949/1504477.
* Waymo Open Dataset: this software and the results obtained with it were made using the Waymo Open Dataset, provided
  by Waymo LLC under the Waymo Dataset License Agreement for Non-Commercial Use, available at waymo.com/open/terms, and
  your access and use of such work are governed by the terms and conditions therein. The car-following pairs:
  Hu, X.; Zheng, Z.; Zhang, X.; Chen, D.; Sun, J. Vehicle Trajectory Data Processed from the Waymo Open Dataset.
  Mendeley Data, Version 2, 2021. doi:10.17632/wfn2c3437n.2.
* OpenACC: Makridis, M.; Mattas, K.; Anesiadou, A.; Ciuffo, B. OpenACC. An open database of car-following experiments
  to study the properties of commercial ACC systems. Transportation Research Part C: Emerging Technologies 2021, 125,
  103047. doi:10.1016/j.trc.2021.103047. Data: European Commission, Joint Research Centre. Open ACC Database.
  doi:10.2905/JRC.KMH3D00.
* FollowNet: Chen, X.; Zhu, M.; Chen, K.; Wang, P.; Lu, H.; Zhong, H.; Han, X.; Wang, X.; Wang, Y. FollowNet: A
  Comprehensive Benchmark for Car-Following Behavior Modeling. Scientific Data 2023, 10, 828.
  doi:10.1038/s41597-023-02718-7.
