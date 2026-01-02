# ==============================================================================
# AVANCERAD UTVÄRDERING MED CLUSTERED BOOTSTRAP & EXACT MATCH
# ==============================================================================

library(tidyverse)
library(knitr)
rm(list = ls())
# Inställningar
ROUNDING_TOLERANCE <- 0.5
N_BOOTSTRAP <- 1000  # Antal resamples för konfidensintervall

# ==============================================================================
# 1. LÄS IN OCH FÖRBERED DATA
# ==============================================================================

message("Läser in och tvättar data...")

# Läs in filer
gold <- read_delim("gold_standard_test.csv", delim = ";", show_col_types = FALSE) %>% mutate(Source = "Gold")
gemini <- read_csv("20251212_154315_gemini_zero-shot_TEST_extractions.csv", show_col_types = FALSE) %>% mutate(Source = "Gemini")
gpt <- read_csv("gpt_zero_shot_extractions.csv", show_col_types = FALSE) %>% mutate(Source = "GPT")
claude <- read_csv("20251215_131614_claude-haiku_zero-shot_TEST_extractions.csv", show_col_types = FALSE) %>% mutate(Source = "Claude")

# Funktion för att skapa ID och numerisk tvätt
prepare_data <- function(df) {
  df %>%
    mutate(ico_id = paste(pmcid, outcome, intervention, comparator, sep = "_")) %>%
    # Spara pmcid separat för bootstrap-clustering
    mutate(pmcid = as.character(pmcid)) %>% 
    mutate(across(c(contains("mean"), contains("standard_deviation"), 
                    contains("events"), contains("group_size")), 
                  ~suppressWarnings(as.numeric(.))))
}

gold_clean <- prepare_data(gold)
models_list <- list(Gemini = prepare_data(gemini), GPT = prepare_data(gpt), Claude = prepare_data(claude))

# Fältdefinitioner
numeric_targets <- c("intervention_events", "intervention_group_size", 
                     "comparator_events", "comparator_group_size", 
                     "intervention_mean", "intervention_standard_deviation", 
                     "comparator_mean", "comparator_standard_deviation")

binary_fields <- c("intervention_events", "intervention_group_size", 
                   "comparator_events", "comparator_group_size")

# ==============================================================================
# 2. DEFINIERA LOGIK FÖR EVALUATION SCHEME (ROW-LEVEL)
# ==============================================================================

# Denna funktion skapar en tabell med PRE-CALCULATED metrics för varje cell.
# Detta gör bootstrappen mycket snabbare.

calculate_row_metrics <- function(gold_df, model_df, model_name) {
  
  # Långt format för att jämföra varje cell individuellt
  g_long <- gold_df %>% 
    select(ico_id, pmcid, outcome_type, all_of(numeric_targets)) %>%
    pivot_longer(cols = all_of(numeric_targets), names_to = "Field", values_to = "Gold_Val")
  
  m_long <- model_df %>% 
    select(ico_id, all_of(numeric_targets)) %>%
    pivot_longer(cols = all_of(numeric_targets), names_to = "Field", values_to = "Model_Val")
  
  joined <- left_join(g_long, m_long, by = c("ico_id", "Field")) %>%
    mutate(Model = model_name) %>%
    mutate(
      # Definitioner enligt texten:
      # TP: Match (Gold finns, Model matchar)
      is_tp = !is.na(Gold_Val) & !is.na(Model_Val) & abs(Gold_Val - Model_Val) <= ROUNDING_TOLERANCE,
      
      # TN: Correctly Unavailable (Båda NA)
      is_tn = is.na(Gold_Val) & is.na(Model_Val),
      
      # FP: Hallucination (Gold är NA, men Model hittade data)
      is_fp = is.na(Gold_Val) & !is.na(Model_Val),
      
      # FN: Missed OR Incorrectly Extracted (Gold finns, Model är NA eller fel värde)
      is_fn = !is.na(Gold_Val) & (!is.na(Model_Val) & abs(Gold_Val - Model_Val) > ROUNDING_TOLERANCE | is.na(Model_Val)),
      
      # Squared Error (endast för RMSE-beräkning på snittet)
      sq_error = ifelse(!is.na(Gold_Val) & !is.na(Model_Val), (Model_Val - Gold_Val)^2, NA)
    )
  
  return(joined)
}

# ==============================================================================
# 3. DEFINIERA LOGIK FÖR EXACT MATCH (ICO-LEVEL)
# ==============================================================================

calculate_exact_match <- function(gold_df, model_df, model_name) {
  # Joina tabellerna bredvid varandra
  joined <- left_join(gold_df, model_df, by = "ico_id", suffix = c(".gold", ".model"))
  
  # Beräkna Exact Match per rad (ICO)
  joined %>%
    rowwise() %>%
    mutate(
      is_exact_match = case_when(
        # Binära utfall: Kräv matchning på 4 fält
        outcome_type.gold == "binary" ~ {
          fields <- binary_fields
          g_vals <- c_across(paste0(fields, ".gold"))
          m_vals <- c_across(paste0(fields, ".model"))
          
          # Hantera NA: Båda måste vara NA, eller ha samma värde
          all( (is.na(g_vals) & is.na(m_vals)) | (!is.na(g_vals) & !is.na(m_vals) & abs(g_vals - m_vals) <= ROUNDING_TOLERANCE) )
        },
        # Kontinuerliga utfall (eller övriga): Kräv matchning på alla 6-8 fält
        TRUE ~ {
          fields <- numeric_targets
          g_vals <- c_across(paste0(fields, ".gold"))
          m_vals <- c_across(paste0(fields, ".model"))
          all( (is.na(g_vals) & is.na(m_vals)) | (!is.na(g_vals) & !is.na(m_vals) & abs(g_vals - m_vals) <= ROUNDING_TOLERANCE) )
        }
      )
    ) %>%
    ungroup() %>%
    transmute(ico_id, pmcid = pmcid.gold, Model = model_name, is_exact_match = as.integer(is_exact_match))
}

# ==============================================================================
# 4. KÖR BERÄKNINGAR OCH AUDIT
# ==============================================================================

message("Beräknar bas-statistik och identifierar fel i Gold Standard...")

# Förbered listor
all_row_metrics <- list()
all_ico_metrics <- list()
audit_results <- list()

for (m_name in names(models_list)) {
  # 1. Row Metrics (Precision, Recall, RMSE data)
  all_row_metrics[[m_name]] <- calculate_row_metrics(gold_clean, models_list[[m_name]], m_name)
  
  # 2. ICO Metrics (Exact Match)
  all_ico_metrics[[m_name]] <- calculate_exact_match(gold_clean, models_list[[m_name]], m_name)
}

combined_row_data <- bind_rows(all_row_metrics)
combined_ico_data <- bind_rows(all_ico_metrics)

# --- AUDIT: Hitta rader där alla modeller är överens mot Gold ---
# Vi använder en pivoterad vy för detta
audit_view <- bind_rows(
  models_list$Gemini %>% select(ico_id, all_of(numeric_targets)) %>% mutate(Model="Gemini"),
  models_list$GPT %>% select(ico_id, all_of(numeric_targets)) %>% mutate(Model="GPT"),
  models_list$Claude %>% select(ico_id, all_of(numeric_targets)) %>% mutate(Model="Claude")
) %>%
  pivot_longer(cols = numeric_targets, names_to = "Field", values_to = "Value") %>%
  pivot_wider(names_from = Model, values_from = Value) %>%
  left_join(gold_clean %>% select(ico_id, all_of(numeric_targets)) %>% 
              pivot_longer(cols=numeric_targets, names_to="Field", values_to="Gold_Value"), 
            by = c("ico_id", "Field")) %>%
  mutate(
    models_agree = (abs(Gemini - GPT) < ROUNDING_TOLERANCE) & (abs(GPT - Claude) < ROUNDING_TOLERANCE) | (is.na(Gemini) & is.na(GPT) & is.na(Claude)),
    conflict_numeric = models_agree & !is.na(Gemini) & !is.na(Gold_Value) & abs(Gemini - Gold_Value) > ROUNDING_TOLERANCE,
    conflict_missing = models_agree & is.na(Gemini) & !is.na(Gold_Value),
    conflict_hallucination = models_agree & !is.na(Gemini) & is.na(Gold_Value)
  ) %>%
  filter(conflict_numeric | conflict_missing | conflict_hallucination) %>%
  mutate(Type = case_when(
    conflict_numeric ~ "Numeric Disagreement (Gold likely wrong)",
    conflict_missing ~ "Models missed it (Hard extraction)",
    conflict_hallucination ~ "Models found it (Gold missed it)"
  ))

# ==============================================================================
# 5. CLUSTERED BOOTSTRAP LOOP
# ==============================================================================

message(paste("Kör Clustered Bootstrap (B =", N_BOOTSTRAP, ")... Detta kan ta en minut."))

perform_bootstrap <- function(data_row, data_ico, pmcids_list) {
  # 1. Sample PMCIDs with replacement
  sampled_pmcids <- sample(pmcids_list, replace = TRUE)
  
  # Skapa en mappningstabell för att snabbt expandera samplet
  # (Detta är snabbare än att filter() i en loop)
  pmcid_freq <- table(sampled_pmcids)
  
  # Filtrera ut data som matchar (vi måste duplicera rader om en PMCID valdes flera gånger)
  # Vi använder left_join mot frekvenstabellen för att duplicera rader
  weights <- data.frame(pmcid = names(pmcid_freq), weight = as.numeric(pmcid_freq))
  
  # --- Metrics Calculation ---
  
  # Row Level (Prec, Rec, F1, RMSE)
  res_row <- data_row %>%
    inner_join(weights, by = "pmcid") %>% # Behåller bara valda PMCIDs och duplicerar
    group_by(Model) %>%
    summarise(
      # Weight multiplierar antalet fall
      TP = sum(is_tp * weight),
      FP = sum(is_fp * weight),
      FN = sum(is_fn * weight),
      Sum_Sq_Err = sum(sq_error * weight, na.rm = TRUE),
      Count_Err = sum(!is.na(sq_error) * weight),
      .groups = "drop"
    )
  
  # ICO Level (Exact Match)
  res_ico <- data_ico %>%
    inner_join(weights, by = "pmcid") %>%
    group_by(Model) %>%
    summarise(
      Total_ICOs = sum(weight),
      Exact_Matches = sum(is_exact_match * weight),
      .groups = "drop"
    )
  
  # Combine
  full_res <- left_join(res_row, res_ico, by = "Model") %>%
    mutate(
      Precision = TP / (TP + FP), # Hallucination metric
      Recall = TP / (TP + FN),    # Extraction accuracy
      F1 = 2 * (Precision * Recall) / (Precision + Recall),
      RMSE = sqrt(Sum_Sq_Err / Count_Err),
      Exact_Match_Rate = Exact_Matches / Total_ICOs
    )
  
  return(full_res %>% select(Model, Precision, Recall, F1, RMSE, Exact_Match_Rate))
}

# Hämta unika PMCIDs
unique_pmcids <- unique(gold_clean$pmcid)

# Kör bootstrap
boot_results <- replicate(N_BOOTSTRAP, 
                          perform_bootstrap(combined_row_data, combined_ico_data, unique_pmcids), 
                          simplify = FALSE) %>% 
  bind_rows()

# ==============================================================================
# 6. SAMMANSTÄLL RESULTAT MED 95% CI
# ==============================================================================

final_stats <- boot_results %>%
  group_by(Model) %>%
  summarise(
    across(c(Precision, Recall, F1, RMSE, Exact_Match_Rate), 
           list(
             mean = ~mean(., na.rm = TRUE),
             lower = ~quantile(., 0.025, na.rm = TRUE),
             upper = ~quantile(., 0.975, na.rm = TRUE)
           ), 
           .names = "{.col}_{.fn}")
  )

# Formatera snygg tabell (Mean [95% CI])
format_ci <- function(m, l, u) {
  sprintf("%.3f [%.3f, %.3f]", m, l, u)
}

display_table <- final_stats %>%
  transmute(
    Model,
    Precision = format_ci(Precision_mean, Precision_lower, Precision_upper),
    Recall = format_ci(Recall_mean, Recall_lower, Recall_upper),
    F1_Score = format_ci(F1_mean, F1_lower, F1_upper),
    RMSE = format_ci(RMSE_mean, RMSE_lower, RMSE_upper),
    Exact_Match = format_ci(Exact_Match_Rate_mean, Exact_Match_Rate_lower, Exact_Match_Rate_upper)
  )

# ==============================================================================
# 7. RAPPORT
# ==============================================================================

cat("\n=======================================================\n")
cat(" FINAL EVALUATION (Clustered Bootstrap, 1000 resamples)\n")
cat("=======================================================\n")
print(kable(display_table, caption = "Model Performance Metrics (Mean [95% CI])"))

cat("\n\n=======================================================\n")
cat(" GOLD STANDARD AUDIT: POTENTIAL ERRORS IN GROUND TRUTH\n")
cat(" (Rows where all 3 models agree perfectly but contradict Gold)\n")
cat("=======================================================\n")

top_suspects <- audit_view %>%
  filter(Type == "Numeric Disagreement (Gold likely wrong)") %>%
  select(ico_id, Field, Gold_Value, Gemini, GPT, Claude, Type) %>%
  head(100)

print(kable(top_suspects))


# För att se hallucinationer (Där modellerna hittade data som saknades i Gold)
hallucination_suspects <- audit_view %>%
  filter(Type == "Models found it (Gold missed it)") %>%
  select(ico_id, Field, Gold_Value, Gemini, GPT, Claude, Type) %>%
  head(10)

if(nrow(hallucination_suspects) > 0) {
  cat("\n\n--- Possible Missing Data in Gold (Models found data where Gold is empty) ---\n")
  print(kable(hallucination_suspects))
}