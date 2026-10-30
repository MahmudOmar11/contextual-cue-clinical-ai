# Robustness of the matched comparison on the 21 clinician-form families.
#
# 1. Extra LLM cue effect with crossed resampling of case families, clinician participants and the 22 unique LLMs
#    (5,000 draws; crossed_resampling.R). Each unique LLM counts once: its rate is averaged over answer orders, then
#    runs, then panel entries. Clinician rates weight each family equally.
# 2. Clinician mixed models with cue effects varying by case family and participant.
#
# Run from the repository root: Rscript code/analysis/04_robustness_matched.R
# Writes results/estimates/04_robustness_matched.csv.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(lme4)
})

root <- local({
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  if (length(f) == 1) normalizePath(file.path(dirname(f), "..", "..")) else getwd()
})
setwd(root)

source(file.path("code", "analysis", "crossed_resampling.R"))

est_dir <- file.path("results", "estimates")
dir.create(est_dir, recursive = TRUE, showWarnings = FALSE)

cases <- read_csv("data/cases.csv", show_col_types = FALSE)
form_families <- sort(cases$family[cases$clinician_form == 1])
repeated_families <- cases$family[cases$repeated_on_clinician_form == 1]

clin <- read_clinicians()
llm <- read_llm_baseline(form_families)

rows <- list()
add <- function(name, r) {
  rows[[length(rows) + 1]] <<- data.frame(estimate = name, value = round(100 * r[["value"]], 1),
                                          ci_low = round(100 * r[["ci_low"]], 1),
                                          ci_high = round(100 * r[["ci_high"]], 1), n = r[["families"]])
}

all_responses <- llm_cells(llm, valid_only = FALSE)
valid_answers <- llm_cells(llm, valid_only = TRUE)

# One fixed seed per analysis; the valid-answer analysis reuses the draws of the main analysis.
main <- crossed(clin, all_responses, seed = 20261928)
add("extra_llm_cue_effect_selection_crossed_pp", main$selection)
add("extra_llm_cue_effect_accuracy_crossed_pp", main$accuracy)
add("extra_llm_cue_effect_selection_crossed_valid_answers_pp",
    crossed(clin, valid_answers, seed = 20261928)$selection)
add("extra_llm_cue_effect_selection_crossed_excluding_medical_students_pp",
    crossed(filter(clin, training_level != "medical_student"), all_responses, seed = 20262928)$selection)
add("extra_llm_cue_effect_selection_crossed_excluding_repeated_families_pp",
    crossed(filter(clin, !family %in% repeated_families), all_responses, seed = 20264928)$selection)

# Clinician mixed models: random intercepts, and cue effects varying by family and participant
md <- clin |> mutate(participant = factor(participant, levels = sort(unique(participant), method = "radix")),
                     family = factor(family))
fit <- function(outcome, rhs) {
  glmer(as.formula(paste(outcome, "~", rhs)), data = md, family = binomial(),
        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000)))
}
specs <- c(random_intercepts = "cue + (1 | family) + (1 | participant)",
           random_cue_slopes = "cue + (1 + cue || family) + (1 + cue || participant)")
for (spec in names(specs)) {
  f <- fit("is_cue_associated", specs[[spec]])
  co <- coef(summary(f))["cue", ]
  rows[[length(rows) + 1]] <- data.frame(
    estimate = paste0("clinician_cue_odds_ratio_selection_", spec),
    value = round(exp(co[["Estimate"]]), 1),
    ci_low = round(exp(co[["Estimate"]] - 1.96 * co[["Std. Error"]]), 1),
    ci_high = round(exp(co[["Estimate"]] + 1.96 * co[["Std. Error"]]), 1), n = nrow(md))
}
f <- suppressMessages(fit("is_intended", specs[["random_cue_slopes"]]))
rows[[length(rows) + 1]] <- data.frame(estimate = "clinician_accuracy_model_random_cue_slopes_singular",
                                       value = as.integer(isSingular(f, tol = 1e-4)), ci_low = NA, ci_high = NA,
                                       n = nrow(md))

out <- bind_rows(rows)
write_csv(out, file.path(est_dir, "04_robustness_matched.csv"), na = "")
print(as.data.frame(out))
