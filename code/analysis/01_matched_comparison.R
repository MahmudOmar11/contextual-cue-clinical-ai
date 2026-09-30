# Clinician participants versus LLMs: response-level rates with Wilson intervals, mixed-effects
# logistic models and the rater-clustered bootstrap difference-in-differences.
# Writes Table 2, Tables S3 to S9 and S11, and results/estimates/01_matched_comparison.csv.
# Run from the repository root: Rscript code/analysis/01_matched_comparison.R

options(scipen = 10)
suppressWarnings(suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(jsonlite)
  library(lme4)
}))

root <- local({
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  if (length(f) == 1) normalizePath(file.path(dirname(f), "..", "..")) else getwd()
})
data_dir <- file.path(root, "data")
tables_dir <- file.path(root, "results", "tables")
estimates_dir <- file.path(root, "results", "estimates")
for (d in c(tables_dir, estimates_dir)) dir.create(d, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------- data
cases <- read_csv(file.path(data_dir, "cases.csv"), show_col_types = FALSE)
matched_families <- sort(cases$family[cases$clinician_form == 1])
repeated_families <- sort(cases$family[cases$repeated_on_clinician_form == 1])

model_info <- fromJSON(file.path(root, "config", "models.json"))$models %>%
  distinct(model_id = openrouter_id, model = name, provider)

clin <- read_csv(file.path(data_dir, "clinician_responses.csv"), na = "",
                 col_types = cols(.default = "c", form = "i", position = "i", family = "i",
                                  answer_order = "i", confidence = "i")) %>%
  left_join(read_csv(file.path(data_dir, "clinician_participants.csv"), show_col_types = FALSE),
            by = "participant") %>%
  mutate(source = "Clinician", rater = participant,
         cue_associated = as.integer(chosen == "cue_associated"),
         intended = as.integer(chosen == "intended"))

models <- read_csv(file.path(data_dir, "model_responses.csv.gz"), na = "", col_select = -raw_output,
                   col_types = cols(.default = "c", run = "i", family = "i", answer_order = "i")) %>%
  left_join(model_info, by = "model_id") %>%
  mutate(source = "LLM", rater = paste(panel, model_id, run, sep = "::"),
         cue_associated = as.integer(selected_role == "cue_associated"),
         intended = as.integer(selected_role == "intended"))

baseline <- models %>% filter(prompt == "baseline")
matched_models <- models %>% filter(family %in% matched_families)
baseline_matched <- matched_models %>% filter(prompt == "baseline")
matched <- bind_rows(clin, baseline_matched)

# Raters are clinician participants and, for LLMs, each run of each model-panel entry. The bootstrap
# resamples raters in sorted order (clinician codes; model runs by panel, model and run), and the mixed
# models use the same order for the rater levels. Model fits use the rows in data-file order.
rater_order <- list(
  Clinician = sort(unique(clin$rater), method = "radix"),
  LLM = sort(unique(baseline_matched$rater), method = "radix")
)

PANEL <- c(core = "Core-20", frontier = "Frontier-3")
PROMPT <- c(baseline = "Baseline", base_rate = "Base-rate prompt", rare_or_serious = "Rare-or-serious prompt")
SELECTION <- "Selection of the cue-associated diagnosis"

# ---------------------------------------------------------------- helpers
wilson <- function(k, n, z = qnorm(0.975)) {
  p <- k / n
  centre <- (p + z^2 / (2 * n)) / (1 + z^2 / n)
  half <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / (1 + z^2 / n)
  cbind(pmax(0, centre - half), pmin(1, centre + half))
}

pct <- function(x) round(100 * x, 1)

with_ci <- function(x) as.vector(t(outer(x, c("", "_ci_low", "_ci_high"), paste0)))

# One row per group: raters, responses and counts, with rates and Wilson intervals, for each version.
rate_table <- function(d, groups) {
  long <- d %>%
    group_by(across(all_of(c(groups, "version")))) %>%
    summarise(raters = n_distinct(rater), n = n(), selection_k = sum(cue_associated),
              accuracy_k = sum(intended), .groups = "drop")
  for (e in c("selection", "accuracy")) {
    ci <- wilson(long[[paste0(e, "_k")]], long$n)
    long[[e]] <- long[[paste0(e, "_k")]] / long$n
    long[[paste0(e, "_ci_low")]] <- ci[, 1]
    long[[paste0(e, "_ci_high")]] <- ci[, 2]
  }
  long %>%
    pivot_wider(names_from = version, names_glue = "{version}_{.value}",
                values_from = c(raters, n, selection_k, accuracy_k, with_ci(c("selection", "accuracy"))))
}

SEL_ACC <- with_ci(c("no_cue_selection", "cue_selection", "no_cue_accuracy", "cue_accuracy"))
ACC_SEL <- with_ci(c("no_cue_accuracy", "cue_accuracy", "no_cue_selection", "cue_selection"))

write_table <- function(x, name) {
  write.csv(x, file.path(tables_dir, paste0(name, ".csv")), row.names = FALSE, na = "")
}

in_order <- function(x, column, levels) x[order(match(x[[column]], levels)), ]

# ---------------------------------------------------------------- Table 2 and Table S3 (matched families)
matched_groups <- bind_rows(
  clin %>% mutate(group = "Clinician participants"),
  baseline_matched %>% mutate(group = "All baseline LLMs"),
  baseline_matched %>% filter(panel == "core") %>% mutate(group = "Core-20 (baseline)"),
  baseline_matched %>% filter(panel == "frontier") %>% mutate(group = "Frontier LLMs")
) %>%
  rate_table("group")

table_2 <- matched_groups %>%
  filter(group != "Core-20 (baseline)") %>%
  in_order("group", c("Clinician participants", "All baseline LLMs", "Frontier LLMs")) %>%
  transmute(group, raters = cue_raters, responses = cue_n + no_cue_n, across(all_of(ACC_SEL), pct))
write_table(table_2, "table_2")

table_s3 <- matched_groups %>%
  mutate(group = recode(group, `Frontier LLMs` = "Frontier-3 (baseline)")) %>%
  in_order("group", c("Clinician participants", "All baseline LLMs", "Core-20 (baseline)",
                      "Frontier-3 (baseline)")) %>%
  transmute(group, across(all_of(SEL_ACC), pct))
write_table(table_s3, "table_s3")

# ---------------------------------------------------------------- Table S4 (prompts, all 100 families)
table_s4 <- models %>%
  rate_table(c("panel", "prompt")) %>%
  arrange(match(panel, names(PANEL)), match(prompt, names(PROMPT))) %>%
  transmute(panel = PANEL[panel], prompt = PROMPT[prompt], across(all_of(SEL_ACC), pct))
write_table(table_s4, "table_s4")

# ---------------------------------------------------------------- Table S5 (frontier LLMs by prompt)
table_s5 <- models %>%
  filter(panel == "frontier") %>%
  rate_table(c("model", "prompt")) %>%
  arrange(match(prompt, names(PROMPT)), model) %>%
  transmute(model, prompt = PROMPT[prompt], across(all_of(SEL_ACC), pct))
write_table(table_s5, "table_s5")

# ---------------------------------------------------------------- Table S6 (per model, matched, baseline)
# Gemini 3.1 Pro pools its Core-20 and Frontier-3 runs.
table_s6 <- baseline_matched %>%
  rate_table(c("model_id", "model", "provider")) %>%
  arrange(desc(cue_selection), model_id) %>%
  transmute(model, provider, across(all_of(with_ci(c("no_cue_selection", "cue_selection", "cue_accuracy"))), pct))
write_table(table_s6, "table_s6")

# ---------------------------------------------------------------- Table S7 (sensitivity analyses)
repeated_label <- paste0("Excluding repeated case families (", paste(repeated_families, collapse = ", "), ")")
sensitivity <- bind_rows(
  clin %>% mutate(analysis = "Primary (matched 21 cases)", group = "Clinician participants"),
  baseline_matched %>% mutate(analysis = "Primary (matched 21 cases)", group = "All baseline LLMs"),
  clin %>% filter(!family %in% repeated_families) %>%
    mutate(analysis = repeated_label, group = "Clinician participants"),
  baseline_matched %>% filter(!family %in% repeated_families) %>%
    mutate(analysis = repeated_label, group = "All baseline LLMs"),
  clin %>% filter(training_level != "medical_student") %>%
    mutate(analysis = "Excluding medical students", group = "Clinician participants"),
  baseline_matched %>% mutate(analysis = "Excluding medical students", group = "All baseline LLMs"),
  clin %>% mutate(analysis = "Frontier-3 comparator only", group = "Clinician participants"),
  baseline_matched %>% filter(panel == "frontier") %>%
    mutate(analysis = "Frontier-3 comparator only", group = "Frontier LLMs")
) %>%
  rate_table(c("analysis", "group"))
table_s7 <- sensitivity %>%
  arrange(match(analysis, c("Primary (matched 21 cases)", repeated_label, "Excluding medical students",
                            "Frontier-3 comparator only")),
          group != "Clinician participants") %>%
  transmute(analysis, group, across(all_of(SEL_ACC), pct))
write_table(table_s7, "table_s7")

# ---------------------------------------------------------------- Table S8 (mixed-effects logistic models)
fit_glmer <- function(formula, d) {
  glmer(formula, data = d, family = binomial(),
        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000)))
}

or_table <- function(fit, terms) {
  s <- coef(summary(fit))
  tibble(term = unname(terms[rownames(s)]),
         estimate = s[, "Estimate"], se = s[, "Std. Error"],
         odds_ratio = exp(s[, "Estimate"]),
         ci_low = exp(s[, "Estimate"] - 1.96 * s[, "Std. Error"]),
         ci_high = exp(s[, "Estimate"] + 1.96 * s[, "Std. Error"]),
         p_value = s[, "Pr(>|z|)"])
}

model_data <- matched %>%
  mutate(source = factor(source, levels = c("Clinician", "LLM")),
         condition = factor(version, levels = c("no_cue", "cue")),
         family_f = factor(family),
         rater_f = factor(rater, levels = unlist(rater_order, use.names = FALSE)))
interaction_terms <- c(`(Intercept)` = "Intercept", sourceLLM = "LLM (vs clinician)",
                       conditioncue = "Cue (vs no cue)", `sourceLLM:conditioncue` = "LLM x cue interaction")
table_s8 <- bind_rows(
  or_table(fit_glmer(cue_associated ~ source * condition + (1 | family_f) + (1 | rater_f), model_data),
           interaction_terms) %>% mutate(outcome = SELECTION),
  or_table(fit_glmer(intended ~ source * condition + (1 | family_f) + (1 | rater_f), model_data),
           interaction_terms) %>% mutate(outcome = "Accuracy")
) %>%
  transmute(outcome, term, estimate = round(estimate, 4), se = round(se, 4),
            odds_ratio = round(odds_ratio, 4), ci_low = round(ci_low, 4), ci_high = round(ci_high, 4),
            p_value = signif(p_value, 3))
write_table(table_s8, "table_s8")

# ---------------------------------------------------------------- Table S9 (rater-clustered bootstrap)
# Raters are resampled with replacement within each source; the extra LLM cue effect is the LLM
# cue-minus-no-cue difference minus the clinician difference, in percentage points.
bootstrap_did <- function(d, outcome, seed, reps = 1000) {
  counts <- d %>%
    group_by(source, rater, version) %>%
    summarise(k = sum(.data[[outcome]]), n = n(), .groups = "drop")
  cells <- lapply(names(rater_order), function(src) {
    get <- function(v, what) {
      x <- counts[counts$source == src & counts$version == v, ]
      x[[what]][match(rater_order[[src]], x$rater)]
    }
    list(k_cue = get("cue", "k"), n_cue = get("cue", "n"), k_no = get("no_cue", "k"), n_no = get("no_cue", "n"))
  })
  names(cells) <- names(rater_order)
  delta <- function(src, w) {
    x <- cells[[src]]
    sum(w * x$k_cue) / sum(w * x$n_cue) - sum(w * x$k_no) / sum(w * x$n_no)
  }
  observed <- delta("LLM", 1) - delta("Clinician", 1)
  set.seed(seed)
  draws <- replicate(reps, {
    w <- lapply(rater_order, function(ids) {
      tabulate(match(sample(ids, size = length(ids), replace = TRUE), ids), nbins = length(ids))
    })
    delta("LLM", w$LLM) - delta("Clinician", w$Clinician)
  })
  tibble(difference = observed, ci_low = unname(quantile(draws, 0.025)),
         ci_high = unname(quantile(draws, 0.975)), bootstrap_mean = mean(draws), resamples = reps)
}

table_s9 <- bind_rows(
  bootstrap_did(matched, "cue_associated", seed = 20260622) %>% mutate(endpoint = SELECTION),
  bootstrap_did(matched, "intended", seed = 20260623) %>% mutate(endpoint = "Accuracy")
) %>%
  transmute(endpoint, difference_pp = pct(difference), ci_low = pct(ci_low), ci_high = pct(ci_high),
            bootstrap_mean_pp = pct(bootstrap_mean), resamples)
write_table(table_s9, "table_s9")

# ---------------------------------------------------------------- Table S11 (confidence, clinicians)
confidence_data <- clin %>%
  mutate(condition = factor(version, levels = c("no_cue", "cue")), family_f = factor(family),
         rater_f = factor(rater, levels = rater_order$Clinician))
confidence_terms <- c(`(Intercept)` = "Intercept", confidence = "Confidence", conditioncue = "Cue (vs no cue)",
                      `confidence:conditioncue` = "Confidence x cue")
table_s11 <- or_table(fit_glmer(cue_associated ~ confidence * condition + (1 | family_f) + (1 | rater_f),
                                confidence_data), confidence_terms) %>%
  transmute(outcome = SELECTION, term, odds_ratio = round(odds_ratio, 4), ci_low = round(ci_low, 4),
            ci_high = round(ci_high, 4), p_value = signif(p_value, 3))
write_table(table_s11, "table_s11")

# ---------------------------------------------------------------- estimates reported in the text
estimate_rows <- function(w, label, versions = c("no_cue", "cue"), endpoints = c("selection", "accuracy")) {
  out <- list()
  for (e in endpoints) for (v in versions) {
    name <- paste0(label, ", ", ifelse(v == "cue", "cue", "no-cue"), " cases: ",
                   ifelse(e == "selection", "selection of the cue-associated diagnosis", "accuracy"))
    out[[length(out) + 1]] <- tibble(
      estimate = c(paste0(name, " (%)"), paste0(name, " (responses)")),
      value = c(pct(w[[paste0(v, "_", e)]]), w[[paste0(v, "_", e, "_k")]]),
      ci_low = c(pct(w[[paste0(v, "_", e, "_ci_low")]]), NA),
      ci_high = c(pct(w[[paste0(v, "_", e, "_ci_high")]]), NA),
      n = w[[paste0(v, "_n")]])
  }
  bind_rows(out)
}

change_rows <- function(w, label) {
  tibble(estimate = paste0(label, ": ", c("cue effect on selection of the cue-associated diagnosis",
                                          "cue effect on accuracy"), " (percentage points)"),
         value = c(pct(w$cue_selection - w$no_cue_selection), pct(w$cue_accuracy - w$no_cue_accuracy)),
         ci_low = NA, ci_high = NA, n = w$cue_n + w$no_cue_n)
}

one <- function(d) rate_table(d %>% mutate(g = 1), "g")
all_clin <- one(clin)
all_baseline <- one(baseline)
all_matched <- one(baseline_matched)
prompt_matched <- rate_table(matched_models, "prompt")
frontier_prompt_matched <- rate_table(matched_models %>% filter(panel == "frontier"), "prompt")
estimates <- bind_rows(
  estimate_rows(all_clin, "Clinician participants, all responses"),
  estimate_rows(all_baseline, "All baseline LLMs, all 100 case families"),
  estimate_rows(all_matched, "All baseline LLMs, matched 21 case families"),
  change_rows(all_matched, "All baseline LLMs, matched 21 case families"),
  change_rows(all_clin, "Clinician participants, matched 21 case families"),
  estimate_rows(prompt_matched %>% filter(prompt == "base_rate"), "All LLMs, base-rate prompt, matched 21 case families"),
  estimate_rows(prompt_matched %>% filter(prompt == "rare_or_serious"),
                "All LLMs, rare-or-serious prompt, matched 21 case families"),
  estimate_rows(frontier_prompt_matched %>% filter(prompt == "base_rate"),
                "Frontier LLMs, base-rate prompt, matched 21 case families", "cue", "selection"),
  estimate_rows(frontier_prompt_matched %>% filter(prompt == "rare_or_serious"),
                "Frontier LLMs, rare-or-serious prompt, matched 21 case families", "cue", "selection"),
  estimate_rows(one(clin %>% filter(training_level != "medical_student")),
                "Clinician participants excluding medical students"),
  estimate_rows(one(clin %>% filter(!family %in% repeated_families)),
                "Clinician participants excluding the repeated case families", endpoints = "selection"),
  tibble(estimate = c("Clinician participants", "Raters among baseline LLMs on the matched case families",
                      "Unique LLMs", "Model-panel entries"),
         value = c(length(rater_order$Clinician), length(rater_order$LLM), n_distinct(models$model_id),
                   nrow(distinct(models, panel, model_id))),
         ci_low = NA, ci_high = NA, n = NA)
)
write.csv(estimates, file.path(estimates_dir, "01_matched_comparison.csv"), row.names = FALSE, na = "")

cat("Wrote Table 2, Tables S3-S9 and S11, and", nrow(estimates), "estimates.\n")
cat(sprintf("Extra LLM cue effect on selection: %.1f (95%% CI %.1f to %.1f) percentage points\n",
            table_s9$difference_pp[1], table_s9$ci_low[1], table_s9$ci_high[1]))
