# NOTICE: data sources and the terms that apply to the derived files

The code, the configurations and the documentation of this repository are under the MIT licence (`LICENSE`). The
run outputs published with them (trained checkpoints, calibrations, per-event metrics, audits, corridor scenarios
and corridor results, tables and figures) were computed from third-party data, and every file derived from a source
remains subject to the terms of that source. The MIT licence does not extend to those files; the terms below are
those of the data providers.

No data set is redistributed here, with one exception that its licence permits: the corridor scenarios hold
trajectories reconstructed from NGSIM (ShareAlike, see below). No event, trajectory or sample of highD or of the
Waymo Open Dataset is in this repository, in its history or in its archived releases.

| Source | Used as | Files derived from it | Terms |
|---|---|---|---|
| **highD** (Krajewski et al., 2018; levelXdata) | the highD car-following events of the FollowNet release (Chen et al., 2023); the raw highD recordings were not used | the checkpoints `runs/<experiment>/follownet_highd/**/model.pt` and every checkpoint fine-tuned from them (the learned models of the runs on `ngsim_i80` and `ngsim_i80_p0`; `init_from` in `metrics.json` names the source run), the per-event metrics `test_events.parquet`, the audits (`stability.json`, `platoon.json`, `certificate.json`, `full_history.json`), `data/calibration/follownet_highd/`, the fold definitions `data/splits/follownet_highd_*.json`, and the tables and figures built from them | Free for non-commercial use only. The data set and modified versions of it may not be distributed; derivative works may be distributed only as abstract representations of the data set (such as trained models) that neither include its data nor allow it to be recovered; no derivative work may be used commercially; every work that uses the data set must reference it (https://levelxdata.com/highd-dataset/). The files listed here are such abstract representations (model weights, fitted parameters, per-event error metrics, aggregate statistics) and are provided for non-commercial research and teaching only. |
| **NGSIM** I-80 and US-101 (U.S. Department of Transportation, Federal Highway Administration; ITS DataHub, doi:10.21949/1504477) | the raw trajectories of data.transportation.gov (data set `8ect-6jqj`), reconstructed by this project (`docs/decisions.md`, D42-D46), and the NGSIM I-80 events of the FollowNet release | the corridor scenarios `runs/corridor/scenarios/**` (the reconstructed trajectories of the study sections `ground_truth.npz`, the vehicle tables `vehicles_truth.npz`, the boundary speed series `boundary.npz`, the SUMO demand `routes.rou.xml`, the ground-truth metrics), the corridor results `runs/corridor/**`, the law tables `runs/corridor/laws/*.npz`, `data/calibration/ngsim_*/` and `data/calibration/follownet_ngsim_i80/`, the checkpoints trained or fine-tuned on `ngsim_i80` and `ngsim_i80_p0`, the per-event metrics `transfer_ngsim_*.parquet` and of the runs on these sets, the fold definitions | The portal publishes the data set under "Creative Commons Attribution, Share Alike 3.0 Unported" (licence id `CC_30_BY_SA`; its Common Core field names CC BY-SA 4.0). These files are adaptations of NGSIM and are shared under **CC BY-SA 4.0** (https://creativecommons.org/licenses/by-sa/4.0/), which the ShareAlike clause of version 3.0 permits for adaptations. Changes made: selection of the study sections and periods, reconstruction and smoothing of the trajectories, conversion to SI units, extraction of car-following events. |
| **Waymo Open Dataset** through the car-following pairs of Hu et al. (2021; Mendeley Data, doi:10.17632/wfn2c3437n.2, CC BY 4.0 on Mendeley) and the Waymo event set of the FollowNet release | evaluation only: transfer of the models trained on highD (experiment E3) and IDM calibration. No model of this repository was trained on Waymo data | the per-event metrics `transfer_waymo.parquet`, `data/calibration/waymo/` and `data/calibration/follownet_waymo/`, the fold definitions `data/splits/*waymo*.json`, and the rows of the tables built from them | The underlying data set is provided by Waymo LLC under the Waymo Dataset License Agreement for Non-Commercial Use (https://waymo.com/open/terms). Works made using it are for non-commercial purposes only and carry the attribution below. |
| **OpenACC** (Makridis et al., 2021; Joint Research Centre of the European Commission) | the car-following campaigns as events (`openacc`, `openacc_acc`, `openacc_human`) and as leader profiles of the platoon test | the checkpoints `runs/e5*/openacc_*/**/model.pt`, the platoon tests `platoon.json` (statistics per profile), `data/calibration/openacc*/`, the fold definitions | CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), with attribution to the authors and the JRC. Changes made are described in `docs/data_contract.md`. |
| **FollowNet** (Chen et al., 2023; figshare collection doi:10.6084/m9.figshare.c.6777810) | the released event sets of HighD, NGSIM I-80 and Waymo | as listed in the rows of the underlying sources | figshare labels the event sets CC0. That label cannot grant more than the underlying sources allow: the HighD events remain under the highD terms, the NGSIM I-80 events under NGSIM's ShareAlike and the Waymo events under the Waymo agreement. |

## What is not in this repository

* The raw data and the extracted events (`data/raw/`, `data/events/` are excluded from git) and the simulated
  trajectories of the corridor runs.
* Samples of the training data inside checkpoints: the checkpoint of the k-nearest-neighbour baseline holds the
  scaler, the number of samples and the seed only (`docs/decisions.md`, D137); the model is re-fitted from the events
  when it is used, which needs the data (level 3 of the README).

## Which file derives from which data

A training run lives in `runs/<experiment>/<data>/<model>/<run>/`: `<data>` names the event set it was trained on,
and `init_from` in its `metrics.json` names the run it was fine-tuned from. A checkpoint fine-tuned on NGSIM I-80
from a model trained on the highD events is subject to the highD terms and to NGSIM's ShareAlike together. The
corridor laws (`runs/corridor/laws/`) list their member runs.

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
  103047. doi:10.1016/j.trc.2021.103047.
* FollowNet: Chen, X.; Zhu, M.; Chen, K.; Wang, P.; Lu, H.; Zhong, H.; Han, X.; Wang, X.; Wang, Y. FollowNet: A
  Comprehensive Benchmark for Car-Following Behavior Modeling. Scientific Data 2023, 10, 828.
  doi:10.1038/s41597-023-02718-7.
