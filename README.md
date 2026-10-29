# Forecast Accuracy vs. Decision Value in Cloud Power Management Under Workload Drift

Code, trained forecasters, and per-episode results for the paper.
The study asks whether a more accurate workload forecast gives a better host-sleep decision in a cloud data centre.
It compares six forecasters with calibrated upper bounds, 16 tuned no-ML rules (6 static pools and 10 rolling percentiles), a reactive controller, an always-on policy, and an oracle.
That gives 55 strategies per cell. The experiments run on four public traces (Alibaba v2018, Azure Functions 2019, Bitbrains fastStorage and Rnd) with host wake-up times of 120, 300, and 600 s, in CloudSim Plus 8.5.7 (18 hosts, 2 VMs per host, load 1.0).

## Repository layout

| Path | Content |
|---|---|
| `cloudsim/` | CloudSim Plus experiment (main results of the paper) |
| `cloudsim/P1Runner.java` | Java driver: runs the 55 strategies and the task streams in CloudSim Plus |
| `cloudsim/p1_inputs2.py` | builds the host-count sequences and task streams; splits the validation part into calibration and selection halves; seeds episodes with CRC-32 of the trace name |
| `cloudsim/run_cell2.sh`, `cells.txt`, `sens_chain.sh` | run one cell, the cell list, and the fleet/load sensitivity runs |
| `cloudsim/p1_analyze2.py`, `p1_boot2.py`, `p1_extra.py`, `p1_paper2.py`, `p1_figs3.py`, `p1_metrics.py` | selection, paired statistics, bootstrap, robustness table, and every table, number macro, and figure of the paper |
| `cloudsim/verify1.py`, `verify2.py`, `verify3.py` | independent recomputation of the headline numbers, weight table, and robustness runs from the raw result files |
| `cloudsim/zoo/<trace>/` | trained forecaster zoo (`zoo.pkl`: scaled series, forecasters, residual data), `accuracy.csv`, `drift.json` |
| `cloudsim/results/` | per-episode CloudSim results (`res_*`), selection-half errors (`selmae_*`), per-cell summaries (`p1_*_strategies.csv`), bootstrap (`p1_boot_*.csv`), `p1_numbers.json` |
| `slot_level/` | slot-level simulator and checks: forecaster training (`fdv_benchmark.py`), rolling origins (`fdv_robust.py`), residual PPO (`train_ppo.py`), static and rolling baselines |
| `data/README.md` | where to get the traces and how to convert them |

## Result files

`results/res_<tag>_<trace>_<wake>.csv` has one row per strategy, part (`val` or `test`), and episode, with makespan, idle and dynamic energy, late tasks, response time, and wake events.
Tags: `fin` (main design: 60 selection and 30 test episodes per strategy; the paper's headline selection uses the first 20 selection episodes), `finsw` (calibration and selection halves swapped),
`finlo` and `finhi` (load 0.6 and 1.3), `finbig` (54 hosts). The last three cover Alibaba and fastStorage at 300 s.
`selmae_*` holds each model's error on the selection half. `p1_fin20_strategies.csv`, `p1_fin60_strategies.csv`, and `p1_finsw20_strategies.csv` give the per-cell selections and paired differences
(primary design, 60 selection episodes, swapped halves).

## Reproduce the tables, numbers, and figures (no simulation needed)

```bash
pip install -r requirements.txt
mkdir work && cp cloudsim/results/* work/ && cd work
export PYTHONWARNINGS=ignore OUT=../out          # tables and figures go to ../out/tables and ../out/figures
R=../cloudsim
python $R/p1_analyze2.py fin 18 2 1.0                  # primary design, V=20
NV_USE=60 python $R/p1_analyze2.py fin 18 2 1.0        # V=60
python $R/p1_analyze2.py finsw 18 2 1.0                # swapped halves
python $R/p1_boot2.py fin 20; python $R/p1_boot2.py fin 60; python $R/p1_boot2.py finsw 20
python $R/p1_extra.py                                  # robustness table
python $R/p1_paper2.py                                 # tables, number macros, Figures 3, 5, 6
python $R/p1_figs3.py                                  # Figures 4 and 7
python $R/verify1.py; python $R/verify2.py; python $R/verify3.py   # independent recomputation
```

The output tables are byte-identical to the ones in the paper, and the summary and bootstrap CSVs are byte-identical to those in `cloudsim/results/`.

## Re-run the CloudSim Plus simulation

Requires Python 3.10+, Java 17+, and Maven.

```bash
cd cloudsim
mvn -f pom.xml dependency:copy-dependencies -DoutputDirectory=libs     # CloudSim Plus 8.5.7 and dependencies
mkdir out && javac -cp "libs/*" -d out P1Runner.java
./run_cell2.sh Alibaba 300 18 2 1.0 fin        # trace, wake (s), hosts, VMs per host, load, tag
# swapped halves: SWAP=1 ./run_cell2.sh ... finsw      fewer selection episodes: NV=20 ./run_cell2.sh ...
```

One cell takes about 14 minutes on one CPU core. `run_cell2.sh` writes `res_<tag>_<trace>_<wake>.csv` and `selmae_<tag>_<trace>_<wake>.csv`.
Episode seeds derive from `zlib.crc32(trace)`, so a re-run regenerates the same episodes and the same host-count sequences. A re-run of Alibaba at 120 s reproduced the shipped result file byte for byte.

## Retrain the forecasters

```bash
cd slot_level
python fdv_benchmark.py --trace ali_cpu.npy --wakes 120 300 600 --out fdv_ali      # zoo, bounds, slot-level strategies
python fdv_robust.py run --trace bb_fast.npy --out rob_bbfast --wakes 300 600     # rolling origins
python train_ppo.py --trace bb_fast.npy ...                                       # residual PPO check
```

`prepare_trace.py` and `prepare_azure_functions.py` convert the raw trace files (see `data/README.md`).

## Notes

* The paper uses CloudSim Plus 8.5.7 (https://cloudsimplus.org). The jars are not redistributed; `pom.xml` fetches them. CloudSim Plus is released under GPL-3.0, so check its licence before you distribute binaries that bundle it.
* The first CloudSim run of the project used Python's per-process string hash for episode seeds and cannot be regenerated, so it is not included and not used in the paper.
* One forecaster zoo is trained per trace, so the intervals in the paper cover test-episode variation. Training variation is covered by the rolling-origin checks in `slot_level/`.
* Task generation is synthetic: the traces set the load shape, and the SLA, energy, and response-time values are properties of the simulated fleet.

## Citation

See `CITATION.cff`. Muhammd Irfan, Institute of Business and Management, Karachi.

## License

MIT for the code in this repository (see `LICENSE`), except where a file says otherwise.
