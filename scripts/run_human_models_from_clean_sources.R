suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(lme4)
  library(openxlsx)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: run_human_models_from_clean_sources.R <matched_long.csv> <responses.csv> <output_dir>")
}
long_path <- args[[1]]
responses_path <- args[[2]]
out_dir <- args[[3]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

safe_message <- function(fit) {
  messages <- fit@optinfo$conv$lme4$messages
  if (is.null(messages)) "none" else paste(messages, collapse = " | ")
}

random_effect_table <- function(fit, model_name) {
  output <- as.data.frame(VarCorr(fit))
  output$model <- model_name
  output[, c("model", "grp", "var1", "var2", "vcov", "sdcor")]
}

diagnostic_row <- function(fit, model_name, family_name, analysis_set = "all_rows") {
  optimizer <- fit@optinfo$optimizer
  if (length(optimizer) == 0 || is.null(optimizer)) optimizer <- NA_character_
  data.frame(
    model = model_name,
    analysis_set = analysis_set,
    family = family_name,
    observations = nobs(fit),
    singular_fit_tolerance_1e_4 = isSingular(fit, tol = 1e-4),
    optimizer = paste(optimizer, collapse = ";"),
    convergence_messages = safe_message(fit),
    max_absolute_gradient = {
      gradient <- fit@optinfo$derivs$gradient
      if (is.null(gradient)) NA_real_ else max(abs(gradient))
    },
    model_matrix_rank = qr(getME(fit, "X"))$rank,
    fixed_effect_columns = ncol(getME(fit, "X")),
    stringsAsFactors = FALSE
  )
}

tidy_mer <- function(fit, model_name, exponentiate = FALSE, analysis_set = "all_rows") {
  coefficients <- coef(summary(fit))
  estimate <- coefficients[, "Estimate"]
  standard_error <- coefficients[, "Std. Error"]
  statistic_col <- intersect(c("z value", "t value"), colnames(coefficients))[[1]]
  output <- data.frame(
    model = model_name,
    analysis_set = analysis_set,
    term = rownames(coefficients),
    estimate = estimate,
    std_error = standard_error,
    statistic = coefficients[, statistic_col],
    ci_low = estimate - 1.96 * standard_error,
    ci_high = estimate + 1.96 * standard_error,
    row.names = NULL,
    stringsAsFactors = FALSE
  )
  if ("Pr(>|z|)" %in% colnames(coefficients)) {
    output$p_value <- coefficients[, "Pr(>|z|)"]
  } else {
    output$p_value <- NA_real_
  }
  if (exponentiate) {
    output$odds_ratio <- exp(output$estimate)
    output$odds_ratio_ci_low <- exp(output$ci_low)
    output$odds_ratio_ci_high <- exp(output$ci_high)
  }
  output
}

long <- read.csv(long_path, check.names = FALSE) |>
  mutate(
    evaluator_id = factor(evaluator_id),
    scenario_id = factor(scenario_id),
    method = factor(method),
    domain = factor(domain)
  )

scene <- long |>
  filter(as.character(method) == "PluralPass APS") |>
  distinct(
    scenario_id, domain, candidate_count,
    model_entropy_from_archived_probabilities, set_size
  ) |>
  rename(corrected_aps_set_size = set_size)

responses <- read.csv(responses_path, check.names = FALSE) |>
  mutate(
    evaluator_id = factor(coach_id),
    scenario_id = factor(as.character(scenario_id)),
    assistance_condition = relevel(factor(assistance_condition), ref = "pluralpass_set"),
    attention_passed = tolower(as.character(attention_check_passed)) == "yes"
  ) |>
  left_join(scene, by = "scenario_id")

coverage_methods <- c(
  "PluralPass APS",
  "PluralPass RAPS lambda=0.005 k=3",
  "same-size HGB set",
  "same-size candidate-wise MLP set"
)

fit_coverage <- function(analysis_set = "all_rows") {
  data <- long |>
    filter(as.character(method) %in% coverage_methods)
  if (analysis_set == "attention_passed") {
    valid_keys <- responses |>
      filter(attention_passed) |>
      distinct(evaluator_id, scenario_id)
    data <- data |>
      inner_join(valid_keys, by = c("evaluator_id", "scenario_id"))
  }
  data <- data |>
    mutate(method = relevel(droplevels(method), ref = "same-size candidate-wise MLP set"))
  rank1 <- glmer(
    rank1_covered ~ method + scale(set_size) +
      scale(model_entropy_from_archived_probabilities) + scale(scenario_order) +
      (1 | evaluator_id) + (1 | scenario_id),
    data = data,
    family = binomial,
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))
  )
  acceptable <- glmer(
    any_acceptable_covered ~ method + scale(set_size) +
      scale(model_entropy_from_archived_probabilities) + scale(scenario_order) +
      (1 | evaluator_id) + (1 | scenario_id),
    data = data,
    family = binomial,
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))
  )
  list(rank1 = rank1, acceptable = acceptable)
}

fit_human <- function(analysis_set = "all_rows") {
  data <- responses
  if (analysis_set == "attention_passed") data <- data |> filter(attention_passed)

  rating_long <- data |>
    select(
      evaluator_id, scenario_id, assistance_condition, scenario_order,
      corrected_aps_set_size, model_entropy_from_archived_probabilities,
      recommendation_reasonableness_1_7, information_gain_1_7,
      intended_use_1_7, explanation_clarity_1_7
    ) |>
    pivot_longer(
      cols = c(
        recommendation_reasonableness_1_7, information_gain_1_7,
        intended_use_1_7, explanation_clarity_1_7
      ),
      names_to = "rating_dimension",
      values_to = "rating"
    ) |>
    filter(!is.na(rating)) |>
    mutate(
      rating_dimension = relevel(
        factor(rating_dimension), ref = "recommendation_reasonableness_1_7"
      ),
      evaluator_scene = interaction(evaluator_id, scenario_id, drop = TRUE)
    )

  rating <- lmer(
    rating ~ assistance_condition * rating_dimension +
      scale(corrected_aps_set_size) + scale(scenario_order) +
      (1 | evaluator_id) + (1 | scenario_id) + (1 | evaluator_scene),
    data = rating_long,
    REML = FALSE,
    control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))
  )

  time_rows <- data |>
    filter(!is.na(stage2_decision_time_seconds), stage2_decision_time_seconds > 0) |>
    mutate(log_stage2_time = log(stage2_decision_time_seconds))
  time <- lmer(
    log_stage2_time ~ assistance_condition + scale(corrected_aps_set_size) +
      scale(model_entropy_from_archived_probabilities) + scale(scenario_order) +
      (1 | evaluator_id) + (1 | scenario_id),
    data = time_rows,
    REML = FALSE,
    control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))
  )

  binary_models <- list()
  for (outcome in c(
    "missing_important_option", "bad_recommended_option",
    "changed_first_choice_after_assistance"
  )) {
    binary_data <- data |>
      mutate(binary_outcome = case_when(
        tolower(as.character(.data[[outcome]])) == "yes" ~ 1,
        tolower(as.character(.data[[outcome]])) == "no" ~ 0,
        TRUE ~ NA_real_
      )) |>
      filter(!is.na(binary_outcome))
    binary_models[[outcome]] <- glmer(
      binary_outcome ~ assistance_condition + scale(corrected_aps_set_size) +
        scale(model_entropy_from_archived_probabilities) + scale(scenario_order) +
        (1 | evaluator_id) + (1 | scenario_id),
      data = binary_data,
      family = binomial,
      control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))
    )
  }
  list(rating = rating, time = time, binary = binary_models)
}

all_coverage <- fit_coverage("all_rows")
valid_coverage <- fit_coverage("attention_passed")
all_human <- fit_human("all_rows")
valid_human <- fit_human("attention_passed")

fixed_effects <- bind_rows(
  tidy_mer(all_coverage$rank1, "rank1_coverage_glmm", TRUE, "all_rows"),
  tidy_mer(valid_coverage$rank1, "rank1_coverage_glmm", TRUE, "attention_passed"),
  tidy_mer(all_coverage$acceptable, "any_acceptable_coverage_glmm", TRUE, "all_rows"),
  tidy_mer(valid_coverage$acceptable, "any_acceptable_coverage_glmm", TRUE, "attention_passed"),
  tidy_mer(all_human$rating, "rating_dimension_lmm", FALSE, "all_rows"),
  tidy_mer(valid_human$rating, "rating_dimension_lmm", FALSE, "attention_passed"),
  tidy_mer(all_human$time, "decision_time_lmm", FALSE, "all_rows"),
  tidy_mer(valid_human$time, "decision_time_lmm", FALSE, "attention_passed"),
  bind_rows(lapply(names(all_human$binary), function(name) {
    tidy_mer(all_human$binary[[name]], paste0(name, "_glmm"), TRUE, "all_rows")
  })),
  bind_rows(lapply(names(valid_human$binary), function(name) {
    tidy_mer(valid_human$binary[[name]], paste0(name, "_glmm"), TRUE, "attention_passed")
  }))
)

all_fits <- c(
  list(
    rank1_coverage_glmm = all_coverage$rank1,
    any_acceptable_coverage_glmm = all_coverage$acceptable,
    rating_dimension_lmm = all_human$rating,
    decision_time_lmm = all_human$time
  ),
  setNames(all_human$binary, paste0(names(all_human$binary), "_glmm"))
)
valid_fits <- c(
  list(
    rank1_coverage_glmm = valid_coverage$rank1,
    any_acceptable_coverage_glmm = valid_coverage$acceptable,
    rating_dimension_lmm = valid_human$rating,
    decision_time_lmm = valid_human$time
  ),
  setNames(valid_human$binary, paste0(names(valid_human$binary), "_glmm"))
)

random_effects <- bind_rows(
  lapply(names(all_fits), function(name) random_effect_table(all_fits[[name]], name) |>
    mutate(analysis_set = "all_rows")),
  lapply(names(valid_fits), function(name) random_effect_table(valid_fits[[name]], name) |>
    mutate(analysis_set = "attention_passed"))
)

diagnostics <- bind_rows(
  lapply(names(all_fits), function(name) {
    family_name <- if (inherits(all_fits[[name]], "glmerMod")) "binomial-logit" else "Gaussian"
    diagnostic_row(all_fits[[name]], name, family_name, "all_rows")
  }),
  lapply(names(valid_fits), function(name) {
    family_name <- if (inherits(valid_fits[[name]], "glmerMod")) "binomial-logit" else "Gaussian"
    diagnostic_row(valid_fits[[name]], name, family_name, "attention_passed")
  })
)

source_definition <- data.frame(
  item = c(
    "human_response_source", "model_entropy_source", "set_size_source",
    "attention_sensitivity", "role_covariate"
  ),
  definition = c(
    responses_path,
    "correct leave-one-domain candidate probabilities",
    "PluralPass APS set size reconstructed from corrected external-fold probabilities",
    "row-level sensitivity; main analysis retains all 800 rows",
    "not included: archived role labels are unique descriptors and cannot estimate experience effects"
  )
)

write.csv(fixed_effects, file.path(out_dir, "human_model_fixed_effects_v2.csv"), row.names = FALSE)
write.csv(random_effects, file.path(out_dir, "human_model_random_effects_v2.csv"), row.names = FALSE)
write.csv(diagnostics, file.path(out_dir, "human_model_diagnostics_v2.csv"), row.names = FALSE)
write.csv(source_definition, file.path(out_dir, "human_model_source_definitions_v2.csv"), row.names = FALSE)

wb <- createWorkbook()
addWorksheet(wb, "fixed_effects"); writeData(wb, "fixed_effects", fixed_effects)
addWorksheet(wb, "random_effects"); writeData(wb, "random_effects", random_effects)
addWorksheet(wb, "diagnostics"); writeData(wb, "diagnostics", diagnostics)
addWorksheet(wb, "source_definitions"); writeData(wb, "source_definitions", source_definition)
saveWorkbook(wb, file.path(out_dir, "human_hierarchical_models_clean_sources_v2.xlsx"), overwrite = TRUE)

cat(file.path(out_dir, "human_hierarchical_models_clean_sources_v2.xlsx"), "\n")
