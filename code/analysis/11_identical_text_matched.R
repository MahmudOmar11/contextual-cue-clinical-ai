# Matched comparison on identical case text: clinician participants against the three frontier LLMs, whose runs all
# received the final wording that the clinician participants saw (the core-panel baseline runs received an earlier
# wording of 20 of the 21 families). Extra LLM cue effect on selection of the cue-associated diagnosis and on accuracy,
# with crossed resampling of case families and clinician participants (5,000 draws; crossed_resampling.R) and the
# frontier LLMs held fixed as a named set.
#
# Run from the repository root: Rscript code/analysis/11_identical_text_matched.R
# Writes results/estimates/11_identical_text_matched.csv.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

root <- local({
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  if (length(f) == 1) normalizePath(file.path(dirname(f), "..", "..")) else getwd()
})
setwd(root)

source(file.path("code", "analysis", "crossed_resampling.R"))

cases <- read_csv("data/cases.csv", show_col_types = FALSE)
form_families <- sort(cases$family[cases$clinician_form == 1])

clin <- read_clinicians()
frontier <- read_llm_baseline(form_families) |> filter(panel == "frontier")
stopifnot(n_distinct(frontier$model_id) == 3, all(frontier$wording == "final"))

res <- crossed(clin, llm_cells(frontier, valid_only = FALSE), seed = 20261930, n_llms = 3L, fixed_llms = TRUE)
out <- bind_rows(lapply(c("selection", "accuracy"), function(outcome) {
  r <- res[[outcome]]
  data.frame(estimate = paste0("extra_llm_cue_effect_", outcome, "_crossed_frontier_llms_identical_text_pp"),
             value = round(100 * r[["value"]], 1), ci_low = round(100 * r[["ci_low"]], 1),
             ci_high = round(100 * r[["ci_high"]], 1), n = r[["families"]])
}))
write_csv(out, file.path("results", "estimates", "11_identical_text_matched.csv"), na = "")
print(as.data.frame(out))
