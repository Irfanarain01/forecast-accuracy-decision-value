# Traces

The raw traces are not redistributed. All four are public:

| Trace | Source | Use in the paper |
|---|---|---|
| Alibaba cluster-trace-v2018 | https://github.com/alibaba/clusterdata | machine CPU usage, 60 s bins (`ali_cpu.npy`, 11,520 steps) |
| Azure Functions 2019 | https://github.com/Azure/AzurePublicDataset | invocations per minute summed over functions (`azf_load.npy`, 20,160 steps) |
| Bitbrains fastStorage | http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains | CPU usage, VM sums (`bb_fast.npy`, 8,640 steps) |
| Bitbrains Rnd | http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains | CPU usage, VM sums (`bb_rnd.npy`, 26,208 steps) |

Conversion scripts: `slot_level/prepare_trace.py` (Alibaba, Bitbrains) and `slot_level/prepare_azure_functions.py` (Azure Functions).

You do not need the raw traces to reproduce the CloudSim Plus results. `cloudsim/zoo/<trace>/zoo.pkl` already holds the scaled
train, validation, and test series and the trained forecasters for each trace, and `accuracy.csv` and `drift.json` hold their error and drift statistics.
You need the raw traces only to retrain the forecasters with `slot_level/fdv_benchmark.py`.
Check the licence of each dataset before redistributing any derived series.
