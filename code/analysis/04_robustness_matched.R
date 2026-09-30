# Robustness of the matched comparison on the 21 clinician-form families.
#
# 1. Extra LLM cue effect with crossed resampling of case families, clinician participants and the 22 unique LLMs
#    (5,000 draws). Each unique LLM counts once: its rate is averaged over answer orders, then runs, then panel
#    entries. Clinician rates weight each family equally.
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

B <- 5000L
est_dir <- file.path("results", "estimates")
dir.create(est_dir, recursive = TRUE, showWarnings = FALSE)

cases <- read_csv("data/cases.csv", show_col_types = FALSE)
form_families <- sort(cases$family[cases$clinician_form == 1])
repeated_families <- cases$family[cases$repeated_on_clinician_form == 1]

# Clinician responses sorted by participant and position; participants are resampled in sorted code order.
clin <- read_csv("data/clinician_responses.csv", show_col_types = FALSE, na = "") |>
  arrange(participant, position) |>
  left_join(read_csv("data/clinician_participants.csv", show_col_types = FALSE), by = "participant") |>
  mutate(cue = as.integer(version == "cue"),
         is_cue_associated = as.integer(chosen == "cue_associated"),
         is_intended = as.integer(chosen == "intended"))
stopifnot(nrow(clin) == 1128, n_distinct(clin$participant) == 47)

# LLM rates per unique LLM, family and version (baseline prompt, 21 families)
llm <- read_csv("data/model_responses.csv.gz", show_col_types = FALSE, na = character(),
                col_select = c(panel, model_id, prompt, run, family, version, selected_role)) |>
  filter(prompt == "baseline", family %in% form_families) |>
  mutate(cue_associated = as.integer(selected_role == "cue_associated"),
         intended = as.integer(selected_role == "intended"),
         valid = as.integer(selected_role != "no_valid_answer"))

llm_cells <- function(valid_only) {
  if (valid_only) {
    # valid answers only: pooled over runs and answer orders within a panel entry
    panel_rates <- llm |>
      group_by(model_id, panel, family, version) |>
      summarise(cue_associated = sum(cue_associated) / sum(valid), intended = sum(intended) / sum(valid),
                .groups = "drop")
  } else {
    # all responses: responses without a valid answer count as neither diagnosis
    panel_rates <- llm |>
      group_by(model_id, panel, run, family, version) |>
      summarise(cue_associated = mean(cue_associated), intended = mean(intended), .groups = "drop") |>
      group_by(model_id, panel, family, version) |>
      summarise(cue_associated = mean(cue_associated), intended = mean(intended), .groups = "drop")
  }
  panel_rates |>
    group_by(model_id, family, version) |>
    summarise(cue_associated = mean(cue_associated, na.rm = TRUE), intended = mean(intended, na.rm = TRUE),
              .groups = "drop")
}

# Participant x family count arrays for each version: responses, cue-associated, intended
make_mats <- function(d) {
  p <- sort(unique(d$participant), method = "radix")
  ca <- sort(unique(d$family))
  n <- length(p)
  k <- length(ca)
  mats <- lapply(0:1, function(z) {
    a <- array(0, c(n, k, 3))
    dd <- d[d$cue == z, ]
    i <- match(dd$participant, p)
    j <- match(dd$family, ca)
    for (r in seq_len(nrow(dd))) {
      a[i[r], j[r], 1] <- a[i[r], j[r], 1] + 1
      a[i[r], j[r], 2] <- a[i[r], j[r], 2] + dd$is_cue_associated[r]
      a[i[r], j[r], 3] <- a[i[r], j[r], 3] + dd$is_intended[r]
    }
    a
  })
  list(m = mats, n = n, k = k, p = p, ca = ca)
}

# Family-weighted clinician rates under participant weights pw and family weights cw
calc <- function(mm, pw, cw, outcome) {
  rates <- sapply(mm$m, function(a) {
    ns <- as.numeric(pw %*% a[, , 1])
    ss <- as.numeric(pw %*% a[, , outcome])
    if (any(ns[cw > 0] == 0)) return(NA_real_)
    sum(cw[cw > 0] * (ss / ns)[cw > 0]) / sum(cw)
  })
  c(no_cue = rates[1], cue = rates[2], delta = rates[2] - rates[1])
}

crossed <- function(d, cells, seed) {
  mm <- make_mats(d)
  zs <- filter(cells, family %in% mm$ca)
  ids <- sort(unique(zs$model_id), method = "radix")
  nm <- length(ids)
  mz <- array(NA_real_, c(nm, mm$k, 2, 2))
  for (r in seq_len(nrow(zs))) {
    mz[match(zs$model_id[r], ids), match(zs$family[r], mm$ca), match(zs$version[r], c("no_cue", "cue")), ] <-
      c(zs$cue_associated[r], zs$intended[r])
  }
  stopifnot(!anyNA(mz), nm == 22)
  set.seed(seed)
  pw <- t(replicate(B, tabulate(sample.int(mm$n, mm$n, replace = TRUE), nbins = mm$n)))
  mw <- t(replicate(B, tabulate(sample.int(nm, nm, replace = TRUE), nbins = nm)))
  cw <- t(replicate(B, tabulate(sample.int(mm$k, mm$k, replace = TRUE), nbins = mm$k)))
  out <- list()
  for (oi in 1:2) {
    md <- mz[, , 2, oi] - mz[, , 1, oi]
    hd <- unname(calc(mm, rep(1, mm$n), rep(1, mm$k), oi + 1)["delta"])
    vals <- vapply(seq_len(B), function(b) {
      h <- calc(mm, pw[b, ], cw[b, ], oi + 1)["delta"]
      m <- sum(as.numeric(mw[b, ] %*% md) * cw[b, ]) / sum(mw[b, ]) / sum(cw[b, ])
      m - h
    }, numeric(1))
    ok <- is.finite(vals)
    stopifnot(sum(ok) >= 0.99 * B)
    ci <- quantile(vals[ok], c(0.025, 0.975), names = FALSE)
    out[[c("selection", "accuracy")[oi]]] <- c(value = mean(md) - hd, ci_low = ci[1], ci_high = ci[2],
                                              participants = mm$n, families = mm$k)
  }
  out
}

rows <- list()
add <- function(name, r) {
  rows[[length(rows) + 1]] <<- data.frame(estimate = name, value = round(100 * r[["value"]], 1),
                                          ci_low = round(100 * r[["ci_low"]], 1),
                                          ci_high = round(100 * r[["ci_high"]], 1), n = r[["families"]])
}

all_responses <- llm_cells(valid_only = FALSE)
valid_answers <- llm_cells(valid_only = TRUE)

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
