library(tidyverse)
library(RNifti)
library(tidymodels)

# Setup ====

setwd("~/Projects/kLat/code/random_censoring/")

scan_len <- tribble(
  ~task, ~scan_len, ~n_scans,
  "ADDT", 116, 2,
  "FPW", 81, 2,
  "sline", 60, 4
)

dice <- function(x, y) {

  nx <- sum(x)
  ny <- sum(y)

  n <- sum(x & y)

  return((2 * n) / (nx + ny))

}

# Get data ====

comparison_files <- list.files("real_fd2/",
                               pattern = "*stat-t_statmap.nii.gz",
                               recursive = TRUE, full.names = TRUE)

all_files <- tibble(f = comparison_files) %>%
  mutate(
    data = map(f, readNifti, internal = TRUE),
    # thresh_t15 = map(data, ~as.array(.x) > 1.5),
    thresh_t20 = map(data, ~as.array(.x) > 2.0),
    bn = basename(f),
  ) %>%
  separate_wider_delim(bn, delim = "_",
                       names = c("sub", "task", "nframes",
                                 "contrast", NA, NA)) %>%
  mutate(
    nframes = as.numeric(str_remove(nframes, "nframes-")),
    nvox = map_int(thresh_t20, sum)
  ) %>%
  select(sub, task, nframes, contrast, nvox, data, starts_with("thresh"))

all_files %>%
  filter(
    nvox == 0
  )

## Get gold standard images; based on task (one per contrast)
#
# golds <- all_files %>%
#   group_by(sub, task) %>%
#   filter(
#     nframes == max(nframes)
#   ) %>%
#   select_all(~str_replace(., "thresh", "gold")) %>%
#   select(-nframes)

gold_files <- list.files("../half_modeling/out_full/",
                         pattern = "*stat-t_statmap.nii.gz",
                         recursive = TRUE, full.names = TRUE)

gold_data0 <- tibble(f = gold_files) %>%
  mutate(
    data = map(f, readNifti, internal = TRUE),
    # thresh_t15 = map(data, ~as.array(.x) > 1.5),
    thresh_gold = map(data, ~as.array(.x) > 2.0),
    bn = basename(f),
  )

gold_data <- gold_data0 %>%
  separate_wider_delim(bn, delim = "_",
                       names = c("sub", "task", "contrast", NA, NA)) %>%
  mutate(
    nvox = map_int(thresh_gold, sum),
    contrast = contrast %>%
      str_replace("Minus", "m") %>%
      str_replace("forward", "forw") %>%
      str_replace("[Rr]everse", "rev") %>%
      str_replace("lenPlusCol", "colpluslen") %>%
      tolower()
  ) %>%
  select(sub, task, contrast, nvox, data, starts_with("thresh"))

filter(gold_data, sub == "sub-kLat058")

gold_data %>%
  filter(
    nvox == 0
  )

comp <- all_files %>%
  left_join(gold_data,
            by = join_by(sub, task, contrast),
            relationship = "many-to-one")

# Calculate overlap ====

comp_dice <- comp %>%
  mutate(
    contrast = str_remove(contrast, "contrast-"),
    is_condition = str_detect(contrast, "m"),
#
#     overlap_15 = map2_dbl(thresh_t15, gold_t15, dice),
    overlap_20 = map2_dbl(thresh_t20, thresh_gold, dice),
  ) %>%
  arrange(desc(overlap_20))

comp_dice2 <- comp_dice %>%
  select(sub, task, nframes, contrast, is_condition,
         starts_with("overlap_")) %>%
  pivot_longer(starts_with("overlap_"), names_to = "t_thresh",
               values_to = "dice") %>%
  mutate(
    t_thresh = 0.1 * (str_remove(t_thresh, "overlap_") %>%
      as.numeric()),
    task = str_remove(task, "task-"),
    contrast = paste(task, str_replace(contrast, "m", "-")),
  ) %>%
  left_join(scan_len, by = join_by(task)) %>%
  group_by(sub, task, contrast) %>%
  mutate(
    total_length = scan_len * n_scans,
    frames_as_prop = nframes / total_length,
    prop_total = nframes / max(nframes)
  )

ggplot(comp_dice2,
       aes(x = nframes, y = dice, color = contrast, alpha = is_condition)) +
  geom_point(aes(shape = is_condition), alpha = 0.4) +
  geom_line(aes(group = interaction(sub, contrast), linetype = is_condition)) +
  geom_hline(yintercept = 0.5, color = "red") +
  scale_x_continuous(sec.axis = sec_axis(transform = ~ 100 * .x / max(.x),
                                         breaks = seq(50, 100, by = 10))) +
  scale_linetype_manual(values = c("dashed", "solid")) +
  scale_alpha_manual(values = c(0.75, 1))+
  facet_wrap(vars(task, sub), scales = "free_x") +
  theme_bw() +
  labs(x = "# frames", y = "Dice overlap")

ggplot(comp_dice2,
       aes(x = frames_as_prop, y = dice, color = contrast,
           alpha = is_condition)) +
  geom_point(aes(shape = is_condition), alpha = 0.4) +
  geom_line(aes(group = interaction(sub, contrast), linetype = is_condition)) +
  geom_hline(yintercept = 0.5, color = "red") +
  scale_x_continuous(limits = c(NA, 1), breaks = seq(0, 1, by = 0.2)) +
  scale_linetype_manual(values = c("dashed", "solid")) +
  scale_alpha_manual(values = c(0.75, 1))+
  facet_wrap(vars(task, sub)) +
  theme_bw() +
  labs(x = "% of total scan", y = "Dice overlap")

png("all_subs_all_contrasts_dice.png", width = 6.5, height = 7, units = "in",
    res = 300)

ggplot(filter(comp_dice2, is_condition),
       aes(x = frames_as_prop, y = dice, color = contrast)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se = FALSE) +
  geom_hline(yintercept = 0.5, color = "red") +
  scale_x_continuous(limits = c(NA, 1), breaks = seq(0, 1, by = 0.2)) +
  facet_grid(cols = vars(task), rows = vars(sub)) +
  theme_bw() +
  labs(x = "% of total scan", y = "Dice overlap",
       title = "All contrasts whole-brain Dice by sub") +
  theme(legend.position = "bottom")

dev.off()

models <- comp_dice2 %>%
  group_by(sub, task, contrast) %>%
  nest() %>%
  filter(
    str_detect(contrast, "-")
  ) %>%
  mutate(
    lm = map(data, ~lm(dice ~ frames_as_prop, data = .x)),
    tidy = map(lm, tidy)
  )

models_yhat <- models %>%
  select(-data, -lm) %>%
  unnest(tidy) %>%
  mutate(
    term = str_replace(term, "[(]Intercept[)]", "intercept")
  ) %>%
  select(-std.error, -statistic, -p.value) %>%
  pivot_wider(names_from = term, values_from = estimate) %>%
  mutate(
    dice50 = (0.5 - intercept) / frames_as_prop,
    dice60 = (0.6 - intercept) / frames_as_prop,
    across(c(intercept, dice50, dice60), ~round(.x, 2))
  ) %>%
  arrange(task, contrast)

models_yhat_summary <- models_yhat %>%
  group_by(contrast) %>%
  summarize(
    min = min(dice50),
    mean = mean(dice50),
    median = median(dice50),
    max = max(dice50),
    q90 = quantile(dice50, probs = 0.9)
  )

ggplot(models_yhat, aes(x = contrast, y = dice50, color = task)) +
  geom_point() +
  geom_hline(yintercept = 1, color = "red") +
  scale_x_discrete(labels = str_replace(models_yhat$contrast, " ", "\n")) +
  theme_bw()
