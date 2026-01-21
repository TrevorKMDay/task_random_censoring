#!/bin/bash

bids=~/Projects/kLat/bids/
deriv=/Volumes/thufir/kLat_fmriprep/

for n in {0..4} ; do

    out="out/out${n}"

    for i in ADDT FPW sline ; do

        mkdir -p "${out}/${i}"
        python random_cens_task.py      \
            ${bids} ${deriv}        \
            ${i} ${i}_contrasts.json "${out}/${i}" \
            kLat004

        echo "Done with rep ${n} task ${i}"

    done


    # break

done