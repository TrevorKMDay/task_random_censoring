library(tidyverse)

setwd("~/Projects/kLat/code/random_censoring/")

# Motion =====

motion <- read_csv("../characterize_motion/run_motion_ver2.csv",
                   show_col_types = FALSE) %>%
  arrange(total_pct_outlier)

motion2 <- motion %>%
  filter(
    (task %in% c("task-ADDT", "task-FPW") & included_runs >= 2) |
      (task == "task-sline" & included_runs >= 4)
  ) %>%
  mutate(
    sub_task = paste(sub, task)
  )

# LIs ====

li_files <- list.files("..", pattern = ".*_LIs_CC-reformatted.csv",
                       recursive = TRUE, full.names = TRUE)

LIs0 <- tibble(f = li_files) %>%
  mutate(
    data = map(f, read_csv, show_col_types = FALSE)
  )

LIs <- LIs0 %>%
  unnest(data) %>%
  select(input_image, inclusive_mask, li_wm) %>%
  separate_wider_delim(input_image, "_",
                       names = c("sub", "task", "contrast", NA, NA, "cc"))

LIs_contrasts <- LIs %>%
  mutate(
    roi = str_extract(inclusive_mask, "roi-[^_]*") %>%
      str_remove("roi-"),
    contrast = str_remove(contrast, "contrast-")
  ) %>%
  filter(
    contrast %in% c("lenMinusCol", "forwardMinusReverse", "facesMinusPlaces",
                    "wordsMinusPlaces")
  ) %>%
  na.omit() %>%
  select(sub, task, roi, contrast, li_wm) %>%
  left_join(
    select(motion, sub, task, total_pct_outlier)
  )

LIsZ <- LIs_contrasts %>%
  group_by(contrast, roi) %>%
  mutate(
    li_wm_Z = scale(li_wm, center = FALSE)[, 1],
    pct_outlier_Z = scale(total_pct_outlier, center = FALSE)[, 1]
  )

ggplot(LIs_contrasts,
       aes(x = total_pct_outlier, y = li_wm, color = contrast)) +
  geom_point() +
  facet_wrap(vars(task, roi)) +
  theme_bw()

# LIs_bilateral <- LIs_contrasts %>%
#   mutate(
#     sub_task = paste(sub, task)
#   ) %>%
#   filter(
#     sub_task %in% motion2$sub_task
#   ) %>%
#   group_by(contrast, roi) %>%
#   filter(
#     abs(li_wm) == min(abs(li_wm))
#   )
#
# write_csv(LIs_bilateral, "subs_task_to_use.csv")

LIs_bilateral <- LIsZ %>%
  mutate(
    euclid = sqrt(li_wm_Z^2 + pct_outlier_Z^2),
    min = euclid == min(euclid)
  )

ggplot(LIs_bilateral,
       aes(x = pct_outlier_Z, y = li_wm_Z, color = contrast)) +
  geom_point(aes(shape = min)) +
  facet_wrap(vars(task, roi)) +
  theme_bw()

LIs_bi_min <- LIs_bilateral %>%
  filter(
    min
  )

write_csv(LIs_bi_min, "subs_task_to_use2.csv")
