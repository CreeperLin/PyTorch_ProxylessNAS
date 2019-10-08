#!/bin/sh
set -e
path=`cd $1; pwd`
prim=("IDT" "AVG" "MAX" "SC3" "SC5" "SC7" "DC3" "DC5" "C")
echo -e "ep. \c"
for p in ${prim[@]}; do
    echo -e "$p \c"
done
echo ""
for i in $(seq -f "%03g" 0 $2);do
    echo -e "$i \c"
    for p in ${prim[@]}; do
        count=`grep -o $p "${path}/gene_$i.gt" | wc -l`
        echo -e "$count \c"
    done
    echo ""
done