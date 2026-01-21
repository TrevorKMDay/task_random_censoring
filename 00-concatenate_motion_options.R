library(tidyverse)

setwd("~/Projects/kLat/code/random_censoring/")

to_copy <- list.files("/Volumes/thufir/kLat_fmriprep/",
                      pattern = "*confounds_timeseries.tsv", recursive = TRUE,
                      full.names = TRUE)

for (i in to_copy) {

  file.copy(i, "confounds/", overwrite = FALSE)

}

conf_files <- list.files("confounds/", "*.tsv", full.names = TRUE)

confs <- tibble(f = conf_files) %>%
  mutate(
    data = map(f, read_tsv, show_col_types = FALSE, na = "n/a",
               .progress = TRUE)
  )

fd <- confs %>%
  mutate(
    fd = map(data, ~ tibble(fd = .x$framewise_displacement,
                            frame = 1:nrow(.x))),
    f = str_remove(f, "confounds/*")
  ) %>%
  select(-data) %>%
  separate_wider_delim(f, "_", names = c("sub", "task", "run", NA, NA)) %>%
  mutate(
    cens1mm = map_dbl(fd, ~ sum(.x$fd > 1, na.rm = TRUE) / nrow(.x))
  )

ggplot(fd, aes(x = cens1mm)) +
  geom_histogram(binwidth = 0.05, boundary = 0) +
  facet_wrap(vars(task)) +
  theme_bw()

sline <- fd %>%
  filter(
    task == "task-sline"
  )

ADDT <- fd %>%
  filter(
    task == "task-ADDT"
  )

FPW <- fd %>%
  filter(
    task == "task-FPW"
  )

widen <- function(x) {

  y <- x %>%
    unnest(fd) %>%
    mutate(
      fd = round(fd, 4)
    )

  z <- y %>%
    pivot_wider(names_from = frame, names_prefix = "frame",
                values_from = fd) %>%
    arrange(cens1mm)

  return(z)

}

sline_wide <- widen(sline)
ADDT_wide <- widen(ADDT)
FPW_wide <- widen(FPW)

dir.create("censoring_files/", showWarnings = FALSE)

write_csv(sline_wide, "censoring_files/task-sline.csv")
write_csv(ADDT_wide, "censoring_files/task-ADDT.csv")
write_csv(FPW_wide, "censoring_files/task-FPW.csv")
