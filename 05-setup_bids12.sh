#!/bin/bash

mkdir -p {bids,derivs}{1,2}/

bids=~/Projects/kLat/bids/
deriv=/Volumes/thufir/kLat_fmriprep/

subs=$(sed 1d subs_task_to_use.csv | awk -F, '{print $1}')

for sub in ${subs} ; do

    mkdir -p {bids,derivs}{1,2}/"${sub}"/func

    for i in 1 2 ; do

        #shellcheck disable=SC2046
        rsync \
            --update \
            $(find ${bids}/"${sub}"/func/ -name "*task-ADDT_run-${i}_*") \
            $(find ${bids}/"${sub}"/func/ -name "*task-FPW_run-${i}_*") \
            "bids${i}/${sub}/func"

        #shellcheck disable=SC2046
        rsync \
            --update \
            $(find ${deriv}/"${sub}"/func/ -name "*task-ADDT_run-${i}_*bold*") \
            $(find ${deriv}/"${sub}"/func/ -name "*task-FPW_run-${i}_*bold*") \
            "derivs${i}/${sub}/func"

    done


    #shellcheck disable=SC2046
    rsync \
        --update \
        $(find ${bids}/"${sub}"/func/ -name "*task-sline_run-[12]_*bold*") \
        "bids1/${sub}/func"

    #shellcheck disable=SC2046
    rsync \
        --update \
        $(find ${bids}/"${sub}"/func/ -name "*task-sline_run-[34]_*bold*") \
        "bids2/${sub}/func"

    #shellcheck disable=SC2046
    rsync \
        --update \
        $(find ${deriv}/"${sub}"/func/ -name "*task-sline_run-[12]_*bold*") \
        "derivs1/${sub}/func"

    #shellcheck disable=SC2046
    rsync \
        --update \
        $(find ${deriv}/"${sub}"/func/ -name "*task-sline_run-[34]_*bold*") \
        "derivs2/${sub}/func"

done