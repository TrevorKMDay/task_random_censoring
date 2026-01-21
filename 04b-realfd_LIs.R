library(tidyverse)

# Setup ====

setwd("~/Projects/kLat/code/random_censoring/")

select <- dplyr::select
source("~/code/R/read_reformatted.R")

scan_len <- tribble(
    ~task, ~scan_len, ~n_scans,
    "ADDT", 116, 2,
    "FPW", 81, 2,
    "sline", 60, 4
  ) %>%
  mutate(
    total_len = scan_len * n_scans
  )

# Load data ====

all_files <- list.files(".", pattern = ".*_realfd2-reformatted.csv")

LIs0 <- tibble(f = all_files) %>%
  mutate(
    data = map(f, read_csv, show_col_types = FALSE),
    roi1 = case_when(
      str_detect(f, "ADDT_frontal") ~ "frontal",
      str_detect(f, "ADDT_temporal") ~ "temporal",
      str_detect(f, "FPW") ~ "VWFA",
      str_detect(f, "sline") ~ "parietal"
    ),
    dat2 = map2(data, roi1,
                ~read_reformatted(.x,
                                  names = c("sub", "task", "nframes",
                                            "contrast", NA, NA),
                                  roi = .y))
  )

gold_files <- list.files("../half_modeling/", pattern = "*_LIs-reformatted.csv",
                         full.names = TRUE)

gold0 <- tibble(f = gold_files) %>%
  mutate(
    data = map(f, read_csv, show_col_types = FALSE),
    roi1 = case_when(
      str_detect(f, "ADDT_frontal") ~ "frontal",
      str_detect(f, "ADDT_temporal") ~ "temporal",
      str_detect(f, "FPW") ~ "VWFA",
      str_detect(f, "sline") ~ "parietal"
    ),
    dat2 = map2(data, roi1,
                ~read_reformatted(.x,
                                  names = c("sub", "task", "contrast",
                                            NA, NA, "rep"),
                                  roi = .y))
  )

gold <- gold0 %>%
  select(dat2) %>%
  unnest(dat2) %>%
  filter(
    rep == "out_full"
  ) %>%
  select(-rep) %>%
  mutate(
    contrast = contrast %>%
      str_replace("m", "-") %>%
      str_replace("forward", "forw") %>%
      str_replace("[Rr]everse", "rev") %>%
      tolower() %>%
      paste(task, .)
  )

# Format data ====

all <- LIs0 %>%
  select(-data) %>%
  unnest(dat2) %>%
  mutate(
    nframes = as.numeric(nframes),
    contrast = paste(task, contrast) %>%
      str_replace("m", "-")
  )

# Get motion maxima ====

subs <- unique(all$sub)

confounds <- list.files("/Volumes/thufir/kLat_fmriprep/",
                        pattern = ".*confounds_timeseries.tsv",
                        recursive = TRUE, full.names = TRUE)

confounds2 <- tibble(f = confounds) %>%
  mutate(
    bn = basename(f)
  ) %>%
  separate_wider_delim(bn, "_", names = c("sub", "task", "run", NA, NA)) %>%
  filter(
    sub %in% paste0("sub-", subs)
  ) %>%
  mutate(
    data = map(f, read_tsv, show_col_types = FALSE),
    nrow = map_int(data, nrow),
    fd = map_int(data, ~sum(.x$framewise_displacement > 1, na.rm = TRUE))
  )

confounds_by_task <- confounds2 %>%
  select(sub, task, run, nrow, fd) %>%
  group_by(sub, task) %>%
  summarize(
    nrow = sum(nrow),
    fd = sum(fd)
  ) %>%
  mutate(
    sub = str_remove(sub, "sub-"),
    task = str_remove(task, "task-"),
    nframes = nrow - fd,
  )

confounds_and_li <- left_join(confounds_by_task, gold)

all2 <- bind_rows(all, confounds_and_li) %>%
  filter(
    # (task == "ADDT" & sub %in% c("kLat058", "kLat071")) |
    #   (task == "FPW" & sub == "kLat004") |
    #   (task == "sline" & sub == "kLat016"),
    contrast != "sline col-len"
  ) %>%
  left_join(scan_len) %>%
  mutate(
    perc = nframes / total_len,
    lateralization = case_when(
      li_wm > 0.2 ~ "right",
      li_wm < -0.2 ~ "left",
      TRUE ~ "bilateral"
    )
  )

# Plots ====

ggplot(all2, aes(x = perc, y = li_wm)) +
  geom_point(alpha = 0.25, shape = 20) +
  geom_smooth(aes(color = contrast), method = "lm", se = FALSE) +
  geom_hline(yintercept = 0, color = "red") +
  facet_wrap(vars(task, roi, sub)) +
  theme_bw() +
  labs(x = "Total frames", y = "LI", title = "Total frames")

png("LI_by_subs-all.png", width = 6.5, height = 7.5, units = "in", res = 300)

ggplot(filter(all2, str_detect(contrast, "-")), aes(x = perc, y = li_wm)) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = -0.2, ymax = 0.2,
           fill = "#F8766D", alpha = 0.5) +
  geom_hline(yintercept = 0, color = "red") +
  geom_point(aes(fill = lateralization), shape = 21, size = 2.5) +
  geom_smooth(aes(color = contrast), method = "lm", se = FALSE) +
  scale_y_continuous(breaks = seq(-1, 1, by = 0.4),
                     minor_breaks =  seq(-1, 1, by = 0.2)) +
  facet_grid(cols = vars(task, roi), rows = vars(sub)) +
  theme_bw() +
  labs(x = "Total frames", y = "LI", title = "Total frames by sub") +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(ncol = 3))

dev.off()

# Error ====

abs_err <- all2 %>%
  select(-nrow, -fd, -f, -roi1) %>%
  group_by(sub, task, contrast, roi) %>%
  mutate(
    err = abs(li_wm - li_wm[which.max(nframes)])
  ) %>%
  arrange(sub, task, nframes, contrast) %>%
  left_join(scan_len) %>%
  mutate(
    percent = nframes / total_len,
    is_condition = str_detect(contrast, "-")
  ) %>%
  filter(
    err != 0
  )

ggplot(abs_err, aes(x = percent, y = err, color = contrast)) +
  geom_point(alpha = 0.5) +
  geom_smooth(aes(linetype = is_condition), method = "lm", se = FALSE) +
  scale_linetype_manual(values = c("dashed", "solid")) +
  scale_x_continuous(limits = c(NA, 1)) +
  facet_wrap(vars(task, roi, sub)) +
  theme_bw() +
  labs(x = "Absolute error", y = "% scan kept")

ggplot(filter(abs_err, str_detect(contrast, "-")),
       aes(x = percent, y = err, color = contrast)) +
  geom_point(alpha = 0.5) +
  geom_smooth(se = FALSE) +
  scale_x_continuous(limits = c(NA, 1)) +
  facet_wrap(vars(task, roi, sub)) +
  theme_bw() +
  labs(y = "Absolute error", x = "% scan kept") +
  theme(legend.position = "bottom")
