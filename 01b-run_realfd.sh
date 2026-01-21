#!/bin/bash

bids=~/Projects/kLat/bids/
deriv=/Volumes/thufir/kLat_fmriprep/

# subs=$(sed 1d subs_task_to_use.csv | awk -F, '{print $1}')

# This runs all tasks for all subs

sed 1d subs_task_to_use2.csv | while IFS=, read -r sub x y ; do

    for t in ADDT FPW sline ; do

        cens_file=censoring_files/task-${t}.csv

        python random_cens_task.py                  \
            ${bids} ${deriv}                        \
            "${t}" "${t}_contrasts.json" "real_fd2/"    \
            "${sub//sub-/}"                         \
            --motion "${cens_file}"

    done

    # break

done

# This runs sub-task pairs

#shellcheck disable=SC2034
# sed 1d subs_task_to_use2.csv | while IFS=, read -r sub t x ; do

#     task=${t//task-/}
#     cens_file=censoring_files/task-${task}.csv

#     python random_cens_task.py                          \
#         ${bids} ${deriv}                                \
#         "${task}" "${task}_contrasts.json" "real_fd2/"  \
#         "${sub//sub-/}"                                 \
#         --motion "${cens_file}" 20

#     # break

# done