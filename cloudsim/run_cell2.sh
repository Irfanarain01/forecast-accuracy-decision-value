#!/bin/sh
# usage: run_cell2.sh trace wake hosts vph rho tag   (calibration/selection split, NV=60)
d=in_$6_$1_$2; python3 p1_inputs2.py $1 $2 $3 $4 $5 $d > /dev/null 2>&1 && java -cp "out:libs/*" P1Runner $3 $4 $2 $d res_$6_$1_$2.csv > /dev/null 2>&1; cp $d/sel_mae.csv selmae_$6_$1_$2.csv; rm -rf $d; echo "done $1 $2 $6 $(date +%T)"
