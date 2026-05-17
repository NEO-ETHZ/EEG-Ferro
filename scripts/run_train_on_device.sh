#!/usr/bin/env bash

run_sweeps () {
    local SCRIPT="$1"

    echo ">>> Running sweeps with $SCRIPT"

    echo "###############################"
    echo "### Running sweeps for all layers"
    echo "###############################"

    # Sweep 1: q = 2, 4
   # for q in 2 4; do
   #     echo "=== q=${q} ==="
   #     python "$SCRIPT" \
   #         --q "$q" 
   # done

#     # Sweep 1: thr = 0.025, 0.05, 0.075
#    for thr in 0.025 0.05 0.075; do
#        echo "=== thr=${thr} ==="
#        python "$SCRIPT" \
#            --thr "$thr" 
#    done

    # Sweep 2: thr_asym = 0.5, 1.0, 2.0
#    for thrasym in 0.5 1.0 2.0; do
#        echo "=== thr_asym=${thrasym} ==="
#        python "$SCRIPT" \
#            --thr_asym "$thrasym"
#    done

    # # Sweep 2: rsd = 0.1, 0.2, 0.5
    for rsd_c2c in 0.1 0.2 0.5; do
         echo "=== rsd=${rsd_c2c} ==="
         python "$SCRIPT" \
             --rsd_c2c "$rsd_c2c"
     done
}

run_sweeps "src/training/train_on_device.py"
