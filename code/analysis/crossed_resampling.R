# Crossed resampling for the comparison of clinician participants with LLMs on the 21 clinician-form families.
# Shared by 04_robustness_matched.R and 11_identical_text_matched.R; sourced after the working directory is set to the
# repository root.
#
# Each unique LLM counts once: its rate is averaged over answer orders, then runs, then panel entries (or, with valid
# answers only, pooled over runs and answer orders within a panel entry). Clinician rates weight each family equally.
# Case families, clinician participants and LLMs are resampled together; a named set of LLMs can be held fixed.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

B <- 5000L

# Clinician responses sorted by participant and position; participants are resampled in sorted code order.
read_clinicians <- function() {
  clin <- read_csv("data/clinician_responses.csv", show_col_types = FALSE, na = "") |>
    arrange(participant, position) |>
    left_join(read_csv("data/clinician_participants.csv", show_col_types = FALSE), by = "participant") |>
    mutate(cue = as.integer(version == "cue"),
           is_cue_associated = as.integer(chosen == "cue_associated"),
           is_intended = as.integer(chosen == "intended"))
  stopifnot(nrow(clin) == 1128, n_distinct(clin$participant) == 47)
  clin
}

# Baseline LLM responses on the given families
read_llm_baseline <- function(families) {
  read_csv("data/model_responses.csv.gz", show_col_types = FALSE, na = character(),
           col_select = c(panel, model_id, prompt, run, family, version, wording, selected_role)) |>
    filter(prompt == "baseline", family %in% families) |>
    mutate(cue_associated = as.integer(selected_role == "cue_associated"),
           intended = as.integer(selected_role == "intended"),
           valid = as.integer(selected_role != "no_valid_answer"))
}

# LLM rates per unique LLM, family and version
llm_cells <- function(llm, valid_only) {
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

# Extra LLM cue effect (LLM cue effect minus the clinician cue effect) on selection of the cue-associated diagnosis
# and on accuracy, with percentile intervals. With fixed_llms, every draw weights the LLMs equally.
crossed <- function(d, cells, seed, n_llms = 22L, fixed_llms = FALSE) {
  mm <- make_mats(d)
  zs <- filter(cells, family %in% mm$ca)
  ids <- sort(unique(zs$model_id), method = "radix")
  nm <- length(ids)
  mz <- array(NA_real_, c(nm, mm$k, 2, 2))
  for (r in seq_len(nrow(zs))) {
    mz[match(zs$model_id[r], ids), match(zs$family[r], mm$ca), match(zs$version[r], c("no_cue", "cue")), ] <-
      c(zs$cue_associated[r], zs$intended[r])
  }
  stopifnot(!anyNA(mz), nm == n_llms)
  set.seed(seed)
  pw <- t(replicate(B, tabulate(sample.int(mm$n, mm$n, replace = TRUE), nbins = mm$n)))
  mw <- if (fixed_llms) matrix(1, B, nm) else t(replicate(B, tabulate(sample.int(nm, nm, replace = TRUE), nbins = nm)))
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
