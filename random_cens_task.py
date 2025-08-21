
import argparse as ap
import glob
import numpy as np
import json
import os
import pandas as pd

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
parser.add_argument("subs", nargs="+",
                    help="Subs (without sub-) to process, at least 1.")

parser.add_argument("--motion", "-m", nargs=2, metavar=("file", "n"),
                    help="file: CSV with real motion exclusions. "
                         "n: Number of FD vectors to sample (includes lowest "
                         "and highest-motion rows).")

args = parser.parse_args()

bids_dataset = args.bids_dir
derivatives_folder = args.derivatives_dir
task_label = args.task
space_label = "MNI152NLin2009cAsym"
sub_labels = args.subs
out_dir = args.out_dir

mot_file = args.motion[0]
mot_n = int(args.motion[1])

contrasts_file = args.contrast_file

def directory_exists(dir):

    if os.path.isdir(dir):

        # Check if there are t maps in the directory

        t_maps = glob.glob(f"{dir}/**/*_stat-t_statmap.nii.gz", recursive=True)
        if len(t_maps) > 0:
            print(f"There are {len(t_maps)} t statmaps in {dir}, delete to "
                  "re-run.")
            return(True)
        else:
            print(f"There are 0 t statmaps in {dir}, running.")
            return(False)

    else:
        return(False)

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

def run_at_extra_mask(model, confounds, events, sample_mask, outdir, sub, task,
                      total_mask=1):

    # print(events)

    if isinstance(total_mask, float):

        # If a proportion is given, randomly subsample frames

        if total_mask < 1:

            imgs = [image.load_img(x) for x in fmri_filenames]
            n_scans = [int(img.header["dim"][4]) for img in imgs]

            frames_to_keep = [round(total_mask * x) for x in n_scans]

            try:
                new_list_of_frames = [sample_sample_mask(x, n) for x, n in
                                      zip(sample_mask, frames_to_keep)]
            except ValueError:
                print(f"Too many frames asked for; skipping")
                return

        elif total_mask == 1:

            print("Proportion = 1.0, doing no changes to sample mask!")
            new_list_of_frames = sample_mask

    elif isinstance(total_mask, list):

        # If a list of frames to keep was given, intersect with the real data
        #   to find all the frames to keep (that way a bad frame was never
        #   included)
        new_list_of_frames = [set(i.tolist()) & set(total_mask)
                              for i in sample_mask]
        new_list_of_frames = [np.array(list(x)) for x in new_list_of_frames]

        total_frames = sum([len(x) for x in new_list_of_frames])

        print()
        print(f"{sub} {task}, total frames: {total_frames}")
        print("#################")

        # if os.path.exists(f"{outdir}/"):
        #     print(f"Output directory {outdir} already exists!")
        #     return()

    # Create and fit the model
    model.fit(
        run_imgs=fmri_filenames,
        events=events,
        confounds=confounds,
        sample_masks=new_list_of_frames
    )

    # Load contrasts

    with open(contrasts_file, 'r') as f:
        contrasts = json.load(f)

    save_glm_to_bids(model=model, contrasts=contrasts, out_dir=outdir,
                        prefix=f"sub-{sub}_task-{task}_nframes-{total_frames}")


for sub in sub_labels:

    print(f"Working on sub {sub} {task_label}")

    out_sub_task=f"{out_dir}/sub-{sub}/task-{task_label}/"

    if directory_exists(f"{out_sub_task}"):
        continue

    # From derivatives folder, find the preprocessed BOLD files

    sub_dir = f"{derivatives_folder}/sub-{sub}/func/"

    fmri_filenames = glob.glob(f"{sub_dir}/" +
                               f"*_task-{task_label}*space-{space_label}*" +
                               "_desc-preproc_bold.nii.gz")

    fmri_filenames.sort()
    # print(fmri_filenames)

    # Create confounds and sample mask from derivatives

    confounds, sample_mask = load_confounds(
        fmri_filenames,
        strategy=["high_pass", "motion", "scrub"],
        motion="basic",

        scrub=0,
        fd_threshold=1.0,
        std_dvars_threshold=1.5
    )

    # Get events from BIDS directory

    sub_bids=f"{bids_dataset}/sub-{sub}/func/"
    event_filenames = glob.glob(f"{sub_bids}/" +
                                f"*_task-{task_label}*_events.tsv")

    event_filenames.sort()
    # print(event_filenames)

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

    if mot_file is None:

        step=0.1
        for i in np.arange(0.5, 1 + step, step):

            i = round(i, 2)
            run_at_extra_mask(model, confounds, event_filenames, sample_mask,
                              f"{out_dir}/sub-{sub}/", sub, task_label,
                              total_mask=i)

    else:

        motion = pd.read_csv(mot_file)
        motion_sorted = motion.sort_values(by="cens1mm")
        info, fd = select_motion(motion, mot_n)

        # print(motion.shape)

        for index, row in fd.iterrows():

            # This is a list of T/F values
            r = row.to_list()
            r_indices = [i for i, x in enumerate(r) if not x]

            # print(sample_mask)

            run_at_extra_mask(model, confounds, event_filenames, sample_mask,
                              f"{out_dir}/sub-{sub}/task-{task_label}",
                              sub, task_label,
                              total_mask=r_indices)

        # Do this last so as to not create directory to check for existence
        info_file = f"{out_dir}/sub-{sub}/task-{task_label}/info.csv"
        info.to_csv(info_file)