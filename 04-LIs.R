library(tidyverse)

setwd("~/Projects/kLat/code/random_censoring/")

sline <- read_csv("sline_LIs-reformatted.csv", show_col_types = FALSE) %>%
  select(input_image, li_wm) %>%
  separate_wider_delim(input_image, delim = "_",
                       names = c("task", "prop", "contrast", NA, NA, "rep"),
                       too_few = "align_start") %>%
  mutate(
    across(c(task, prop, contrast, rep), ~str_remove(., "^[^-]*-")),
    prop = as.numeric(prop),
    rep = replace_na(rep, "0"),
    roi = "parietal"
  ) %>%
  filter(
    contrast != "line"
  )

addt_frontal <- read_csv("ADDT_frontal_LIs-reformatted.csv",
                         show_col_types = FALSE) %>%
  select(input_image, li_wm) %>%
  separate_wider_delim(input_image, delim = "_",
                       names = c("task", "prop", "contrast", NA, NA, "rep"),
                       too_few = "align_start") %>%
  mutate(
    across(c(task, prop, contrast, rep), ~str_remove(., "^[^-]*-")),
    prop = as.numeric(prop),
    rep = replace_na(rep, "0"),
    roi = "frontal"
  )

addt_temporal <- read_csv("ADDT_temporal_LIs-reformatted.csv",
                          show_col_types = FALSE) %>%
  select(input_image, li_wm) %>%
  separate_wider_delim(input_image, delim = "_",
                       names = c("task", "prop", "contrast", NA, NA, "rep"),
                       too_few = "align_start") %>%
  mutate(
    across(c(task, prop, contrast, rep), ~str_remove(., "^[^-]*-")),
    prop = as.numeric(prop),
    rep = replace_na(rep, "0"),
    roi = "temporal"
  )

fpw <- read_csv("FPW_LIs-reformatted.csv", show_col_types = FALSE) %>%
  select(input_image, li_wm) %>%
  separate_wider_delim(input_image, delim = "_",
                       names = c("task", "prop", "contrast", NA, NA, "rep"),
                       too_few = "align_start") %>%
  mutate(
    across(c(task, prop, contrast, rep), ~str_remove(., "^[^-]*-")),
    prop = as.numeric(prop),
    rep = replace_na(rep, "0"),
    roi = "vwfa"
  )

all <- bind_rows(sline, addt_frontal, addt_temporal, fpw) %>%
  mutate(
    li_wm = -1 * li_wm
  )

ggplot(all, aes(x = prop, y = abs(li_wm))) +
  geom_point(aes(color = rep), alpha = 0.5) +
  geom_smooth(aes(linetype = roi)) +
  facet_wrap(vars(task, contrast)) +
  theme_bw() +
  labs(x = "Proportion of scan kept", y = "Abs(LI)",
       title = "Absolute value of LI over proportion")

ggplot(all, aes(x = prop, y = li_wm)) +
  geom_point(aes(color = rep), alpha = 0.5) +
  geom_smooth(aes(linetype = roi)) +
  facet_wrap(vars(task, contrast)) +
  theme_bw() +
  labs(x = "Proportion of scan kept", y = "LI")

all_wide <- all %>%
  group_by(task, prop, contrast, roi) %>%
  filter(
    prop < 1
  ) %>%
  summarize(
    m = mean(li_wm, na.rm = TRUE),
    sd = sd(li_wm, na.rm = TRUE),
    sdz = sd / abs(m)
  )

ggplot(all_wide, aes(x = prop, y = sd)) +
  geom_point(alpha = 0.5) +
  geom_smooth(aes(linetype = roi)) +
  geom_hline(yintercept = 0, color = "red") +
  facet_wrap(vars(task, contrast)) +
  theme_bw() +
  labs(title = "SD of estimates (n=5)")

ggplot(all_wide, aes(x = prop, y = sdz)) +
  geom_point(alpha = 0.5) +
  geom_smooth(aes(linetype = roi)) +
  facet_wrap(vars(task, contrast), scales = "free") +
  theme_bw() +
  labs(title = "SD of estimates / mean estimate")

library(tidymodels)

all_models <- all %>%
  group_by(task, contrast, roi) %>%
  nest() %>%
  mutate(
    model = map(data, ~lm(li_wm ~ prop + rep, data = .)),
    m = map(model, tidy)
  )

all_models_tidy <- all_models %>%
  select(task, contrast, roi, m) %>%
  unnest(m) %>%
  filter(
    term == "prop"
  ) %>%
  mutate(
    across(where(is.numeric), ~round(.x, 4))
  )

all_models_tidy$p.adj <- round(p.adjust(all_models_tidy$p.value, "fdr"), 4)

all_models_tidy %>%
  filter(
    p.value < .1
  )

# Real FD ====

files <- list.files(pattern = ".*_LIs_realfd-reformatted.csv")

read_reformatted <- function(refmt, names, roi = NA) {

  x <- refmt %>%
    select(input_image, li_wm) %>%
    separate_wider_delim(input_image, delim = "_", names = names,
                         too_few = "align_start") %>%
    mutate(
      across(everything(), ~str_remove(., "^[^-]*-")),
      roi = roi
    )

  return(x)

}

realfd_LIs0 <- tibble(f = files) %>%
  mutate(
    roi1 = case_when(
      str_detect(f, "ADDT_frontal") ~ "frontal",
      str_detect(f, "ADDT_temporal") ~ "temporal",
      str_detect(f, "FPW") ~ "VWFA",
      str_detect(f, "sline") ~ "parietal",
    ),
    data = map2(f, roi1,
                ~read_csv(.x, show_col_types = FALSE) %>%
                  read_reformatted(.,
                                   c("sub", "task", "nframes", "contrast",
                                     NA, NA),
                                   .y)
               )
  )

scan_len <- tribble(
  ~task, ~scan_len, ~n_scans,
  "ADDT", 116, 2,
  "FPW", 81, 2,
  "sline", 60, 4
)

realfd_LIs <- realfd_LIs0 %>%
  unnest(data) %>%
  select(-roi1) %>%
  mutate(
    across(c(nframes, li_wm), as.numeric)
  ) %>%
  left_join(scan_len) %>%
  group_by(sub, task, contrast, roi) %>%
  mutate(
    total_len = scan_len * n_scans,
    prop = nframes / total_len,

    best_li = li_wm[which.max(nframes)],
    err = li_wm - best_li

  )

ggplot(realfd_LIs, aes(x = prop, y = li_wm, color = sub)) +
  geom_line() +
  geom_point() +
  facet_wrap(vars(contrast, roi)) +
  theme_bw()

ggplot(filter(realfd_LIs, str_detect(contrast, "m")),
       aes(x = prop, y = li_wm, color = sub)) +
  geom_line() +
  geom_point() +
  facet_wrap(vars(contrast, roi)) +
  theme_bw()

ggplot(filter(realfd_LIs, str_detect(contrast, "m")),
       aes(x = prop, y = err, color = sub)) +
  geom_line() +
  geom_point() +
  facet_wrap(vars(contrast, roi)) +
  theme_bw()
