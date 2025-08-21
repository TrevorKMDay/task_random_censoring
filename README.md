# Random censoring of frames for task analysis

# Interfance

    usage: random_cens_task.py [-h] [--motion file] [--fd FD] bids_dir
        derivatives_dir task contrast_file out_dir cens_pct subs [subs ...]

    positional arguments:
    bids_dir              BIDS source directory.
    derivatives_dir       fMRIPREP outputs directory.
    task                  Name of task to process, without 'task-'.
    contrast_file         JSON file with contrasts listed.
    out_dir               Directory to save the individual maps to.
    cens_pct              Censoring percentage, (0, 1].
    subs                  Subs (without sub-) to process, at least 1.

    options:
    -h, --help            show this help message and exit
    --motion file, -m file
                            CSV with real motion exclusions
    --fd FD               FD threshold, in mm (default: 1 mm)

The BIDS and fMRIPREP derivatives directory are always required. The script
works on one `task` at a time, for which a `contrast_file.json` needs to be
created:

    {
        "forw": "forward",
        "rev": "reverse",
        "forwMrev": "forward-reverse"
    }

(Nilearn will do its own adjustment of the contrast names on top of whatever
is set here, e.g. `forwMrev` => `forwmrev`. The behavior is different
between Nilearn modules, so I just set it to what I want where I need it.)

## censoring_files

These are examples of real framewise displacement vectors for each task run.
Columns are `sub`, `task`, `run`, `cens1mm` (the censoring rate at FD > 1 mm),
and then *n* frame columns, where *n* is the length of the task run.

