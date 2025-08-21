
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
from nilearn.glm.first_level import make_first_level_design_matrix, \
    FirstLevelModel
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
space_label = "MNI152NLin2009cAsym"
sub_labels = args.subs
out_dir = args.out_dir
contrasts_file = args.contrast_file

cens_pct = args.cens_pct
fd_thresh = args.fd

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

def make_design_matrix(event_file, image_file):

    img = image.load_img(image_file)
    n_scans = int(img.header["dim"][4])
    tr = img.header["pixdim"][4]

    frame_times = (np.arange(n_scans) * tr)

    events = make_first_level_design_matrix(
        frame_times=frame_times,
        events=event_file,
        hrf_model="spm",
        drift_model=None
    )

    return(events)

def sample_sample_mask(sample_mask, n):

    index = np.random.choice(sample_mask.shape[0], n, replace=False)
    return(sample_mask[index])

def run_at_percentage(model, confounds, events, sample_mask, outdir, sub, task,
                      mask_pct=1):

    # If a proportion is given, randomly subsample frames
    infix="randomframes"

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
            return(False)

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
    print(f"  Using:        {new_length} ({total_frames}) = {p_used}%")

    # Check that this hasn't been done already
    prefix=f"sub-{sub}_task-{task}_{infix}-{total_frames}"

    prefix_files = glob.glob(f"{out_sub_task}/{prefix}_*.nii.gz")

    if len(prefix_files) > 0:

        print(f"Error:   Found {len(prefix_files)} files with the prefix "
            f"{prefix} in {out_sub_task}, not recreating!")

        return(False)

    else:

        # Create and fit the model
        model.fit(
            run_imgs=fmri_filenames,
            events=events,
            confounds=confounds,
            sample_masks=new_list_of_frames
        )

        with open(contrasts_file, 'r') as f:
            contrasts = json.load(f)

        save_glm_to_bids(model=model, contrasts=contrasts, out_dir=outdir,
                         prefix=prefix)

        return(True)


def run_using_vector(model, confounds, events, sample_mask, outdir, sub, task,
                      total_mask=1):

        infix="maskedframes"

        # If a list of frames to keep was given, intersect with the real data
        #   to find all the frames to keep (that way a bad frame was never
        #   included)
        new_list_of_frames = [set(i.tolist()) & set(total_mask)
                              for i in sample_mask]
        new_list_of_frames = [np.array(list(x)) for x in new_list_of_frames]

        total_frames = sum([len(x) for x in new_list_of_frames])

        print()
        print(f"{sub} {task}, total frames: {total_frames}")


        # Load contrasts


        return(True)

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

    # Create design matrix (HRF) based on event file

    # events = [make_design_matrix(e, f) for f, e in
    #           zip(fmri_filenames, event_filenames)]

    # Create the first level model

    model = FirstLevelModel(
        t_r=3.0,
        slice_time_ref=0.5,
        hrf_model="spm",

        smoothing_fwhm=5.0,
        drift_model=None,
        minimize_memory=False
    )

    with tempfile.TemporaryDirectory() as tmpdirname:

        print(f"Info:    Working in temp directory {tmpdirname}")

        if mot_file is None:

            print(f"Info:    Sampling {100 * cens_pct}% of good frames.")

            run_ok = run_at_percentage(model, confounds, event_filenames,
                                       good_volumes, tmpdirname,
                                       sub, task_label, cens_pct)


        # Only copy files if the run was complete (returns True/False)
        if run_ok:

            statmaps = glob.glob(f"{tmpdirname}/*_stat-t_*")

            [shutil.copy2(s, out_sub_task) for s in statmaps]
            print(f"Info:    Copied stat-t maps to {out_sub_task}")