#!/usr/bin/env bash

run_sweeps () {
    local SCRIPT="$1"

    echo ">>> Running sweeps with $SCRIPT"

    echo "###############################"
    echo "### Running sweeps for all layers"
    echo "###############################"

    for x in 2.0 4.0; do
        # Sweep 1: thr = 0.0125, 0.025, 0.05, 0.1
        # for thr in 0.025 0.05 0.075; do
        #     echo "=== thr=${thr} ==="
        python "$SCRIPT" \
            --x "$x" \
            --retuning_epochs 4 # or 10
        # done

        # # Sweep 2: thr_asym = 0.5, 1.0, 2.0
        # for thrasym in 0.5 1.0 2.0; do
        #     echo "=== thr_asym=${thrasym} ==="
        #     python "$SCRIPT" \
        #         --thr_asym "$thrasym" \
        #         --x "$x" 
        # done

        # # Sweep 3: rsd_c2c = 0.5, 1.0, 2.0
        # for rsd in 0.1 0.2 0.5; do
        #     echo "=== rsd=${rsd} ==="
        #     python "$SCRIPT" \
        #         --rsd_c2c "$rsd" \
        #         --x "$x" 
        # done
    done
}

run_sweeps "src/training/retuning.py"
