
import argparse as ap
import glob
import numpy as np
import json
import os
import pandas as pd
import tempfile
import shutil

from pathlib import Path

# Nilearn imports
from nilearn.glm.first_level import FirstLevelModel
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.interfaces.bids import save_glm_to_bids
from nilearn.image import image

from random import sample

from select_motion import select_motion

parser = ap.ArgumentParser()

parser.add_argument("bids_dir", type=str,
                    help="BIDS source directory.")
parser.add_argument("derivatives_dir", type=str,
                    help="fMRIPREP outputs directory.")
parser.add_argument("task",
                    help="Name of task to process, without 'task-'.")
parser.add_argument("contrast_file",
                    help="JSON file with contrasts listed.")
parser.add_argument("out_dir", type=str,
                    help="Directory to save the individual maps to.")
parser.add_argument("cens_pct", type=float,
                    help="Censoring percentage: (0, 1].")
parser.add_argument("subs", nargs="+",
                    help="Subs (without sub-) to process, at least 1.")

parser.add_argument("--motion", "-m", nargs=1, metavar=("file"),
                    help="CSV with real motion exclusions")

parser.add_argument("--fd", type=float, default=1,
                    help="FD threshold, in mm (default: 1 mm)")

args = parser.parse_args()

bids_dataset = args.bids_dir
derivatives_folder = args.derivatives_dir
task_label = args.task
space_label = "MNI152NLin2009cAsym" # TO DO: allow to change this
sub_labels = args.subs
out_dir = args.out_dir
contrasts_file = args.contrast_file

cens_pct = args.cens_pct
fd_thresh = args.fd

# CHECK INPUTS =====

assert cens_pct > 0, "Censoring percentage must be > 0"
assert cens_pct <= 1, "Censoring percentage cannot be > 1."

# Get file of real motion exclusions
if args.motion is not None:
    mot_file = args.motion[0]
    infix = "maskedframes"
else:
    mot_file = None
    infix = "randomframes"

# Information based on censoring percentage set to 1.
if cens_pct == 1:
    print("Warning: Censoring percentage was set to 100%: "
          f"This still censors frames at FD > {fd_thresh}, and sets {infix} to "
          f"'_{infix}-all_'.")

# FUNCTIONS =====

def sample_sample_mask(sample_mask, n):

    index = np.random.choice(sample_mask.shape[0], n, replace=False)
    return(sample_mask[index])

def run_at_percentage(model, confounds, events, sample_mask, sub, task,
                      mask_pct=1):

    # If a proportion is given, randomly subsample frames

    imgs = [image.load_img(x) for x in fmri_filenames]
    n_scans = [int(img.header["dim"][4]) for img in imgs]

    frames_to_keep = [round(mask_pct * x) for x in n_scans]
    # Make sure the same number of frames are extracted from each
    # run - not sure why it ever wouldn't be
    assert len(set(frames_to_keep)) == 1, "Mismatched frame lengths"

    p_scans = [round(mask_pct * n) for n in n_scans]

    if mask_pct < 1:

        try:
            new_list_of_frames = [sample_sample_mask(x, n) for x, n in
                                    zip(sample_mask, frames_to_keep)]
        except ValueError:
            print(f"Error:   Requested {100 * mask_pct}% of {n_scans}, "
                    "but not enough good frames!")
            return(None)

    elif mask_pct == 1:

        new_list_of_frames = sample_mask
        total_frames = "all"

    total_length = sum(n_scans)
    new_length = [len(x) for x in new_list_of_frames]
    frames_used = sum(new_length)
    p_used = round(100 * frames_used / total_length)

    # Create key-value value for frames
    total_frames = frames_used if mask_pct < 1 else f"{frames_used}all"

    print(f"Info:")
    print(f"  Total frames: {n_scans}")
    print(f"  % requested:  {round(mask_pct * 100)}% / {p_scans}")
    print(f"  Using:        {new_length} (total: {total_frames}) = {p_used}%")

    # Check that this hasn't been done already
    infix="randomframes"
    prefix=f"sub-{sub}_task-{task}_{infix}-{total_frames}"
    prefix_files = glob.glob(f"{out_sub_task}/{prefix}_*.nii.gz")

    if len(prefix_files) > 0:

        print(f"Error:   Found {len(prefix_files)} files with the prefix "
            f"{prefix} in {out_sub_task}, not recreating!")

        return(None)

    # Create and fit the model
    model.fit(
        run_imgs=fmri_filenames,
        events=events,
        confounds=confounds,
        sample_masks=new_list_of_frames
    )

    return(model, prefix)


def run_using_vector(model, confounds, events, sample_mask, sub, task,
                     frames_to_keep):

    imgs = [image.load_img(x) for x in fmri_filenames]
    n_scans = [int(img.header["dim"][4]) for img in imgs]
    total_length = sum(n_scans)

    # This is the actual % of KEPT frames
    mask_pct = len(frames_to_keep) / n_scans[0]
    p_scans = [round(mask_pct * n) for n in n_scans]

    # This intersects the new list with the list of good frames so that
    #   only good frames are always modeled, which means the actual included
    #   % will be less than or equal to the requested amount.
    new_list_of_frames = [np.array(list(set(frames_to_keep) & set(x)))
                          for x in sample_mask]

    new_length = [len(x) for x in new_list_of_frames]
    total_frames = sum(new_length)
    p_used = round(100 * total_frames / total_length)

    print(f"Info:")
    print(f"  Total frames: {n_scans}")
    print(f"  % requested:  {round(mask_pct * 100)}% / {p_scans}")
    print(f"  Using:        {new_length} (total: {total_frames}) = {p_used}%")

    # Check that this hasn't been done already
    infix="maskedframes"
    prefix=f"sub-{sub}_task-{task}_{infix}-{total_frames}"
    prefix_files = glob.glob(f"{out_sub_task}/{prefix}_*.nii.gz")

    if len(prefix_files) > 0:

        print(f"Error:   Found {len(prefix_files)} files with the prefix "
              f"{prefix} in {out_sub_task}, not recreating!")

        return(None)

    # Create and fit the model
    model.fit(
        run_imgs=fmri_filenames,
        events=events,
        confounds=confounds,
        sample_masks=new_list_of_frames
    )

    return(model, prefix)

# MAIN LOOP =====

for sub in sub_labels:

    print(f"Info:    Working on sub {sub} {task_label}")

    out_sub_task=f"{out_dir}/sub-{sub}/task-{task_label}/"
    os.makedirs(out_sub_task, exist_ok=True)

    # From derivatives folder, find the preprocessed BOLD files

    sub_dir = f"{derivatives_folder}/sub-{sub}/func/"

    fmri_filenames = glob.glob(f"{sub_dir}/" +
                               f"*_task-{task_label}*space-{space_label}*" +
                               "_desc-preproc_bold.nii.gz")

    fmri_filenames.sort()

    # Create confounds and sample mask from derivatives

    # sample_mask is a LIST OF GOOD VOLUMES:
    #   https://nilearn.github.io/dev/modules/generated/nilearn.interfaces.fmriprep.load_confounds.html
    confounds, good_volumes = load_confounds(
        fmri_filenames,
        strategy=["high_pass", "motion", "scrub"],
        motion="basic",

        scrub=0,
        fd_threshold=fd_thresh,
        std_dvars_threshold=1.5
    )

    # Get events from BIDS directory

    sub_bids=f"{bids_dataset}/sub-{sub}/func/"
    event_filenames = glob.glob(f"{sub_bids}/" +
                                f"*_task-{task_label}*_events.tsv")

    event_filenames.sort()

    # Create the first level model
    model = FirstLevelModel(
        t_r=3.0,
        slice_time_ref=0.5,
        hrf_model="spm",

        smoothing_fwhm=5.0,
        drift_model=None,
        minimize_memory=False

    )

    if mot_file is not None:

        motion = pd.read_csv(mot_file)

        motion_sorted = motion.sort_values(by="cens1mm")
        inclusion = [1 - x for x in motion_sorted["cens1mm"]]

        cens_pct_range = [round(cens_pct - 0.02, 2), round(cens_pct + 0.02, 2)]

        rows_in_range = [x >= cens_pct_range[0] and x <= cens_pct_range[1] for
                         x in inclusion]

        if sum(rows_in_range) == 0:

            print(f"Error:    No rows in range {cens_pct_range} to choose "
                    "from!")

            continue

        else:

            print(f"Info:    {sum(rows_in_range)} in range "
                    f"{cens_pct_range} to choose from! Randomly choosing 1.")

            motion_rows = motion_sorted.iloc[rows_in_range]
            motion_row = motion_rows.sample(n = 1)

            info_cols = ["sub", "task", "run", "cens1mm"]

            # Keep all frames where FD < threshold
            fd = motion_row.drop(info_cols, axis=1).values.tolist()[0]

            # Replace NaNs with 0 so they pass
            fd = [f if not np.isnan(f) else 0 for f in fd]
            frames_to_keep = [ix for ix, f in enumerate(fd) if f < fd_thresh]

            print("Info:    Using a randomly selected vector at about "
                    f"{cens_pct}")

            fitted_model, prefix = run_using_vector(model, confounds,
                                                    event_filenames,
                                                    good_volumes, sub,
                                                    task_label,
                                                    frames_to_keep)

    else:

        print(f"Info:    Sampling {100 * cens_pct}% of good frames.")
        fitted_model, prefix = run_at_percentage(model, confounds,
                                                event_filenames,
                                                good_volumes, sub, task_label,
                                                cens_pct)

    # Only execute model and copy files if the run was complete
    if fitted_model is not None:

        with open(contrasts_file, 'r') as f:
            contrasts = json.load(f)

        with tempfile.TemporaryDirectory() as tmpdirname:

            print(f"Info:    Working in temp directory {tmpdirname}")

            save_glm_to_bids(model=fitted_model, contrasts=contrasts,
                             out_dir=tmpdirname, prefix=prefix)

            # Find and copy files from temp dir to output dir
            statmaps = glob.glob(f"{tmpdirname}/*_stat-t_*")
            [shutil.copy2(s, out_sub_task) for s in statmaps]
            print(f"Info:    Copied stat-t maps to {out_sub_task}")