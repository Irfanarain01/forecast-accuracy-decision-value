#!/bin/sh
while [ "$(grep -c '^done' log_fin.txt)" -lt 24 ]; do sleep 30; done
export NV=20; unset SWAP
{ echo "Alibaba 300 18 2 0.6 finlo"; echo "fastStorage 300 18 2 0.6 finlo"; echo "Alibaba 300 18 2 1.3 finhi"; echo "fastStorage 300 18 2 1.3 finhi"; echo "Alibaba 300 54 2 1.0 finbig"; echo "fastStorage 300 54 2 1.0 finbig"; } | xargs -P2 -L1 sh -c './run_cell2.sh $0 $1 $2 $3 $4 $5'
