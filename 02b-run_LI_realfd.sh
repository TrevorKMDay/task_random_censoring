#!/bin/bash

if [ ${#} -eq 3 ] ; then

    src=${1}
    dst=${2}
    sfx=${3}

else

    echo "Usage: ${0} src/ [for LI dir]/ suffix"
    exit 1

fi

LIt=~/code/LItoolbox/

rois=/Users/tkmd/Projects/kLat/data/rois/

sline_ROI=${rois}/WFU/roi-BA073940_src-WFU_dil-3_hemi-LR_ROI.nii
FPW_ROI=${rois}/SG25/roi-VWFA_src-SG25_space-WFU_hemi-LR.nii

BA44445_ROI=${rois}/WFU/roi-BA4445_src-WFU_dil-3_hemi-LR_ROI.nii
BA22_ROI=${rois}/WFU/roi-BA22_src-WFU_dil-3_hemi-LR_ROI.nii

# Make *.nii files


echo "Starting copy ..."

files=$(find "${src}/" -name "*_stat-t_statmap.nii.gz")
mkdir -p "${dst}/"

for f in ${files} ; do

    rsync --ignore-existing "${f}" "${dst}/$(basename "${f}")"

done


echo "Gunzipping ..."
yes n | gunzip -k "${dst}"/*.nii.gz 1> /dev/null 2> /dev/null

echo "Starting LI calc ..."l

${LIt}/run_LItoolbox_seq.sh                     \
    ${sline_ROI} '-5' "sline_LIs_${sfx}.tsv"    \
    "${dst}"/*_task-sline_*.nii

${LIt}/run_LItoolbox_seq.sh                     \
    ${FPW_ROI} '-5' "FPW_LIs_${sfx}.tsv"        \
    "${dst}"/*_task-FPW_*.nii

${LIt}/run_LItoolbox_seq.sh                         \
    ${BA44445_ROI} '-5' "ADDT_frontal_LIs_${sfx}.tsv" \
    "${dst}"/*_task-ADDT_*.nii

${LIt}/run_LItoolbox_seq.sh \
    ${BA22_ROI} '-5' "ADDT_temporal_LIs_${sfx}.tsv" \
    "${dst}"/*_task-ADDT_*.nii

# Cleanup
rm -f LI_r_*.nii LI_{boot,masking}.ps LI_input*.{hdr,img}

# Reformat output
Rscript ${LIt}/reformat_LI_output.R ./*_"LIs_${sfx}.tsv"