import argparse as ap
import pandas as pd

from select_motion import select_motion

parser = ap.ArgumentParser()

parser.add_argument("motion_file",
                    help="file: CSV with real motion exclusions.")

parser.add_argument("n", default=10,
                    help="Number of FD vectors to sample (includes lowest "
                         "and highest-motion rows).")

args = parser.parse_args()

motion_file = args.motion_file
motion_n = int(args.n)


motion = pd.read_csv(motion_file)
motion_sorted = motion.sort_values(by="cens1mm")

print(f"Loaded {len(motion)} samples.")
print(f"Censoring: {round(100 * min(motion.cens1mm), 1)}%-"
      f"{100 * round(max(motion.cens1mm), 2)}%")

info, fd = select_motion(motion, motion_n)