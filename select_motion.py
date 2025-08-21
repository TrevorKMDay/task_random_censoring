import pandas as pd

def select_motion(motion, size, fd_thresh=1, verbose=False):

    # Create quantiles
    motion["bin"] = pd.qcut(motion["cens1mm"], q=size, duplicates="drop")

    if verbose:
        print(motion["bin"].value_counts())

    rows = motion.groupby("bin", observed=True).sample(n=1)

    print(f"Selected {len(rows)} censoring vectors")

    info_cols = ["sub", "task", "run", "cens1mm", "bin"]
    info = rows[info_cols]

    if verbose:
        print(info)

    fd = rows.drop(info_cols, axis=1)
    fd2 = fd > fd_thresh

    return(info, fd2)