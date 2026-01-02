# ==============================================================================
# SPECIALIZED SCRIPT: JACCARD COMPLETENESS, CONSENSUS & ERROR ANALYSIS
# ==============================================================================

library(tidyverse)
library(knitr)
library(stargazer)
library(dplyr)

rm(list = ls())

# --- CONFIGURATION ---
ROUNDING_TOLERANCE <- 0.00001 # Absolute tolerance
RELATIVE_TOLERANCE <- 0.01    # 1% Relative tolerance for Mean/SD

# ==============================================================================
# 1. DATA LOADING & PREPARATION
# ==============================================================================

message("Loading and cleaning data...")

# Load raw data
gold_raw <- read_delim("gold_standard_clean.csv", delim = ",", show_col_types = FALSE) %>% mutate(Source = "Gold")
gemini   <- read_csv("20251212_154315_gemini_zero-shot_TEST_extractions.csv", show_col_types = FALSE) %>% mutate(Source = "Gemini")
gpt      <- read_csv("gpt_zero_shot_extractions.csv", show_col_types = FALSE) %>% mutate(Source = "GPT")
claude   <- read_csv("20251215_131614_claude-haiku_zero-shot_TEST_extractions.csv", show_col_types = FALSE) %>% mutate(Source = "Claude")

# Filter IDs and select TEST split
remove_ids <- c(2681019, 5244530, 57750, 1216327, 3276927, 
                3580751, 3751573, 4450164, 5498715, 5771543)

gold_clean <- gold_raw %>% 
  filter(!pmcid %in% remove_ids) %>% 
  filter(split == "TEST") %>%
  select(c(3:15, 24)) %>% 
  mutate(ico_id = paste(pmcid, outcome, intervention, comparator, sep = "_")) %>%
  mutate(pmcid = as.character(pmcid)) %>% 
  mutate(across(c(contains("mean"), contains("standard_deviation"), 
                  contains("events"), contains("group_size")), 
                ~suppressWarnings(as.numeric(.))))

# Prepare models
prepare_data <- function(df) {
  df %>%
    mutate(ico_id = paste(pmcid, outcome, intervention, comparator, sep = "_")) %>%
    mutate(across(c(contains("mean"), contains("standard_deviation"), 
                    contains("events"), contains("group_size")), 
                  ~suppressWarnings(as.numeric(.))))
}

models_list <- list(Gemini = prepare_data(gemini), GPT = prepare_data(gpt), Claude = prepare_data(claude))

# Field Definitions
numeric_targets <- c("intervention_events", "intervention_group_size", 
                     "comparator_events", "comparator_group_size",
                     "intervention_mean", "intervention_standard_deviation", 
                     "comparator_mean", "comparator_standard_deviation")

# ==============================================================================
# 2. MATCHING LOGIC (REQUIRED FOR JACCARD)
# ==============================================================================

is_match <- function(gold, model, field_name) {
  diff <- abs(gold - model)
  match_abs <- diff <= ROUNDING_TOLERANCE
  is_relative_field <- grepl("mean|deviation", field_name, ignore.case = TRUE)
  match_rel <- ifelse(is_relative_field & gold != 0, 
                      (diff / abs(gold)) <= RELATIVE_TOLERANCE, FALSE)
  return(match_abs | match_rel)
}

# ==============================================================================
# 3. JACCARD CALCULATION FUNCTION
# ==============================================================================

calculate_numeric_jaccard <- function(gold_df, model_df, model_name) {
  joined <- left_join(gold_df, model_df, by = "ico_id", suffix = c(".gold", ".model")) %>% 
    mutate(intersection_count = 0, union_count = 0)
  
  for (field in numeric_targets) {
    col_g <- paste0(field, ".gold"); col_m <- paste0(field, ".model")
    
    joined <- joined %>% mutate(
      val_g = .data[[col_g]], val_m = .data[[col_m]],
      has_g = !is.na(val_g), has_m = !is.na(val_m),
      
      # Union: Data exists in either
      is_in_union = has_g | has_m,
      
      # Intersection: Data exists in both AND matches tolerance
      is_in_intersection = has_g & has_m & is_match(val_g, val_m, field),
      
      union_count = union_count + as.integer(is_in_union),
      intersection_count = intersection_count + as.integer(is_in_intersection)
    )
  }
  
  joined %>% 
    mutate(
      Jaccard_Index = ifelse(union_count == 0, 1, intersection_count / union_count),
      Model = model_name
    ) %>%
    select(ico_id, pmcid = pmcid.gold, Model, Jaccard_Index)
}

# ==============================================================================
# 4. EXECUTE EVALUATIONS
# ==============================================================================

# --- EVALUATION 1: COMPLETENESS (Model vs Gold) ---
jaccard_vs_gold <- bind_rows(
  calculate_numeric_jaccard(gold_clean, models_list$Gemini, "Gemini"),
  calculate_numeric_jaccard(gold_clean, models_list$GPT, "GPT"),
  calculate_numeric_jaccard(gold_clean, models_list$Claude, "Claude")
)
table_completeness <- jaccard_vs_gold %>% group_by(Model) %>% 
  summarise(Mean_Jaccard = mean(Jaccard_Index, na.rm = TRUE))

# --- EVALUATION 2: CONSENSUS (Model vs Model) ---
jaccard_consensus <- bind_rows(
  calculate_numeric_jaccard(models_list$Gemini, models_list$GPT, "Gemini vs GPT"),
  calculate_numeric_jaccard(models_list$GPT, models_list$Claude, "GPT vs Claude"),
  calculate_numeric_jaccard(models_list$Gemini, models_list$Claude, "Gemini vs Claude")
)
table_consensus <- jaccard_consensus %>% group_by(Model) %>% 
  summarise(Mean_Agreement = mean(Jaccard_Index, na.rm = TRUE))

# --- EVALUATION 3: ERROR ANALYSIS ---
analyze_errors <- function() {
  # Prepare wide data for comparison
  full_data <- gold_clean %>% select(ico_id, pmcid, all_of(numeric_targets)) %>% 
    pivot_longer(cols=all_of(numeric_targets), names_to="Field", values_to="Gold") %>%
    left_join(models_list$Gemini %>% select(ico_id, all_of(numeric_targets)) %>% pivot_longer(cols=all_of(numeric_targets), names_to="Field", values_to="Gemini"), by=c("ico_id","Field")) %>%
    left_join(models_list$GPT %>% select(ico_id, all_of(numeric_targets)) %>% pivot_longer(cols=all_of(numeric_targets), names_to="Field", values_to="GPT"), by=c("ico_id","Field")) %>%
    left_join(models_list$Claude %>% select(ico_id, all_of(numeric_targets)) %>% pivot_longer(cols=all_of(numeric_targets), names_to="Field", values_to="Claude"), by=c("ico_id","Field"))
  
  full_data %>% rowwise() %>% mutate(
    Gemini_Correct = is_match(Gold, Gemini, Field), 
    GPT_Correct = is_match(Gold, GPT, Field), 
    Claude_Correct = is_match(Gold, Claude, Field),
    
    # Consensus Check (Do models agree with each other?)
    Models_Agree_Numeric = !is.na(Gemini) & !is.na(GPT) & !is.na(Claude) & is_match(Gemini, GPT, Field) & is_match(GPT, Claude, Field),
    Models_Agree_Missing = is.na(Gemini) & is.na(GPT) & is.na(Claude),
    Models_Consensus = Models_Agree_Numeric | Models_Agree_Missing,
    
    Any_Error = !(Gemini_Correct & GPT_Correct & Claude_Correct)
  ) %>% filter(Any_Error) %>% ungroup() %>% mutate(
    Error_Type = case_when(
      Models_Consensus & !Gemini_Correct ~ "Systematic Error (Check Gold)",
      (Gemini_Correct + GPT_Correct + Claude_Correct) == 2 ~ "Single Model Failure",
      (Gemini_Correct + GPT_Correct + Claude_Correct) == 1 ~ "Hard Case (2 failed)",
      TRUE ~ "Chaotic Failure"
    )
  )
}

errors_df <- analyze_errors()
table_error_dist <- errors_df %>% count(Error_Type, sort=TRUE) %>% mutate(Percentage = sprintf("%.1f%%", n/sum(n)*100))

# ==============================================================================
# 5. GENERATE OUTPUTS (CONSOLE + LATEX)
# ==============================================================================

# --- Table 1 ---
cat("\n\n")
print(kable(table_completeness, caption = "Table 1: Assessment of Extraction Completeness (Jaccard Index vs. Gold Standard)"))
stargazer(as.data.frame(table_completeness), type = "latex", summary = FALSE, header = FALSE, rownames = FALSE,
          title = "Assessment of Extraction Completeness (Jaccard Index vs. Gold Standard)")

# --- Table 2 ---
cat("\n\n")
print(kable(table_consensus, caption = "Table 2: Inter-Model Consensus Assessment (Pairwise Jaccard Similarity)"))
stargazer(as.data.frame(table_consensus), type = "latex", summary = FALSE, header = FALSE, rownames = FALSE,
          title = "Inter-Model Consensus Assessment (Pairwise Jaccard Similarity)")

# --- Table 3 ---
cat("\n\n")
print(kable(table_error_dist, caption = "Table 3: Taxonomy of Extraction Errors"))
stargazer(as.data.frame(table_error_dist), type = "latex", summary = FALSE, header = FALSE, rownames = FALSE,
          title = "Taxonomy of Extraction Errors")