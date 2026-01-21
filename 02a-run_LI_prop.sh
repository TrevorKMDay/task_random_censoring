#!/bin/bash

LIt=~/code/LItoolbox/

rois=/Users/tkmd/Projects/kLat/data/rois/

sline_ROI=${rois}/WFU/roi-BA073940_src-WFU_dil-3_hemi-LR_ROI.nii
FPW_ROI=${rois}/SG25/roi-VWFA_src-SG25_space-WFU_hemi-LR.nii

BA44445_ROI=${rois}/WFU/roi-BA4445_src-WFU_dil-3_hemi-LR_ROI.nii
BA22_ROI=${rois}/WFU/roi-BA22_src-WFU_dil-3_hemi-LR_ROI.nii

# Make *.nii files

mkdir -p for_li/

echo "Starting copy"

for p in $(seq 0.5 0.05 0.95) ; do

    for i in {0..4} ; do

        files=$(find "out/out${i}"/*/"prop-${p}" -name "*_stat-t_statmap.nii.gz")

        for f in ${files} ; do

            rsync --ignore-existing "${f}" \
                "for_li/$(basename "${f}" .nii.gz)_rep-${i}.nii.gz"

        done

    done

done

# Save a little time by only copying prop-1 files once
rsync --ignore-existing out/out0/*/prop-1.0/*_stat-t_statmap.nii.gz for_li/

echo "Gunzipping ..."
yes n | gunzip -k for_li/*.nii.gz 1> /dev/null

${LIt}/run_LItoolbox_seq.sh \
    ${sline_ROI} '-5' sline_LIs.tsv \
    for_li/task-sline_*.nii

${LIt}/run_LItoolbox_seq.sh \
    ${FPW_ROI} '-5' FPW_LIs.tsv \
    for_li/task-FPW_*.nii

${LIt}/run_LItoolbox_seq.sh \
    ${BA44445_ROI} '-5' ADDT_frontal_LIs.tsv \
    for_li/task-ADDT_*.nii

${LIt}/run_LItoolbox_seq.sh \
    ${BA22_ROI} '-5' ADDT_temporal_LIs.tsv \
    for_li/task-ADDT_*.nii

rm -f LI_r_*.nii LI_{boot,masking}.ps LI_input*.{hdr,img}

Rscript ${LIt}/reformat_LI_output.R ./*_LIs.tsv