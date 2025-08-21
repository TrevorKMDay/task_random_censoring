#!/bin/bash

bids=~/Projects/kLat/bids/
deriv=/Volumes/thufir/kLat_fmriprep/

t=ADDT

#shellcheck disable=SC2034
sed 1d subs_task_to_use2.csv | while IFS=, read -r sub x y ; do

    cens_file=censoring_files/task-${t}.csv

    python random_cens_task.py                      \
        ${bids} ${deriv}                            \
        "${t}" "${t}_contrasts.json" "test/" 0.8  \
        "${sub//sub-/}"     \
        --motion "${cens_file}"

    break

done