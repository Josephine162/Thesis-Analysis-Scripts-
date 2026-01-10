# EEGSurvey_IllustrativeModels_original.R
# Purpose: Illustrative mixed-effects analyses for theory chapter
# Data: ASMR_EEG_survey.xlsx (not included)
# Models:
#   - Logistic GLMM predicting ASMR (Yes/No) from pleasantness and arousal
#   - Linear mixed models predicting tingle intensity on ASMR-positive trials
# Random effects: participant and sound
# Note: Script contains exploratory analyses and multiple plotting iterations.
# A cleaned, modular version will be added in r/theory_illustrative/.

library(tidyverse)
install.packages("lme4")
library(readxl)

df <- read_xlsx("Data/ASMR_EEG_survey.xlsx")

df <- df %>% 
  rename(participant = 'P number')

library(dplyr)
library(tidyr)

# 1. Check the original column names
names(df)

# 2. Pivot from wide to long
df_long <- df %>%
  pivot_longer(
    cols = matches("^\\d"),  
    names_to = c("sound", "question"),
    names_pattern = "(\\d+)\\s+(.*)",
    values_to = "value"
  )

# 3. Inspect how 'question' looks now
unique(df_long$question)

# 4. Create a 'measure' column
df_long2 <- df_long %>%
  mutate(
    measure = case_when(
      grepl("did you", question, ignore.case = TRUE) ~ "asmr",
      grepl("pleasantness_1", question, ignore.case = TRUE) ~ "pleasantness_rating",
      grepl("affect grid_1_x", question, ignore.case = TRUE) ~ "pleasantness_grid",
      grepl("affect grid_1_y", question, ignore.case = TRUE) ~ "arousal_grid",
      TRUE ~ question  # fallback
    )
  )

# 5. Pivot so each measure is its own column
df_final <- df_long2 %>%
  select(participant, sound, measure, value) %>%
  pivot_wider(names_from = measure, values_from = value)

# 6. Rename these new columns
df_final <- df_final %>%
  rename(
    asmr_experience    = asmr,
    pleasantness_scale = pleasantness_rating,
    pleasantness_affectgrid = pleasantness_grid,
    arousal_affectgrid = arousal_grid
  )

# 7. Convert asmr from text to numeric/factor, etc.
df_final <- df_final %>%
  mutate(
    asmr_experience = case_when(
      asmr_experience == "Yes" ~ 1,
      asmr_experience == "No"  ~ 0,
      TRUE ~ NA_real_
    ),
    asmr_experience = factor(asmr_experience, levels = c(0, 1))
  )


df_final <- df_final[-(1:13), ]

library(lme4)

df_final$arousal_affectgrid <- as.numeric(df_final$arousal_affectgrid)

# Example 1: Predict AROUSAL from ASMR
m_arousal <- lmer(
  arousal_affectgrid ~ asmr_experience +
    (1 | participant) +
    (1 | sound),
  data = df_final
)

summary(m_arousal)

df_final$pleasantness_affectgrid <- as.numeric(df_final$pleasantness_affectgrid)

# Example 2: Predict PLEASANTNESS from ASMR
m_pleasantness <- lmer(
  pleasantness_affectgrid ~ asmr_experience +
    (1 | participant) +
    (1 | sound),
  data = df_final
)

summary(m_pleasantness)

#mean pleasantness vs affect plot for each of the 13 sounds 
library(dplyr)
library(ggplot2)

# df_final has one row per participant × sound
# Columns we need: sound (factor or number), pleasantness_affectgrid, arousal_affectgrid

# 1. compute the mean rating for each sound (N = 13)
df_means <- df_final %>% 
  group_by(sound) %>% 
  summarise(
    mean_pleasant = mean(pleasantness_affectgrid, na.rm = TRUE),
    mean_arousal  = mean(arousal_affectgrid,    na.rm = TRUE)
  )

# 2. basic scatter-plot
ggplot(df_means, aes(x = mean_pleasant, y = mean_arousal)) +
  geom_point(size = 3) +
  geom_text(aes(label = sound), vjust = -0.8) +     # optional: label each dot with the sound-ID
  geom_smooth(method = "lm", se = FALSE, colour = "grey40", linetype = "dashed") +  # regression line
  labs(
    x = "Mean pleasantness (valence) rating",
    y = "Mean arousal rating",
    title = "Arousal versus Pleasantness across the 13 sounds"
  ) +
  theme_minimal()

#violin plots to show spread for each sound on arousal and pleasantness
library(ggplot2)
library(ggplot2)

plot_split <- df_final %>% 
  mutate(asmr_label = ifelse(asmr_experience == 1,
                             "ASMR trials", "No-ASMR trials")) %>% 
  pivot_longer(cols = c(pleasantness_affectgrid,
                        arousal_affectgrid),
               names_to  = "dimension",
               values_to = "rating") %>% 
  ggplot(aes(x = factor(sound),
             y = rating,
             fill = asmr_label)) +
  geom_violin(scale = "width", trim = TRUE, alpha = .6) +
  geom_boxplot(width = .1, fill = "white", outlier.shape = NA) +
  facet_grid(dimension ~ asmr_label,    # 4 panels: rows = P/A, cols = ASMR/No
             labeller = labeller(
               dimension = c(pleasantness_affectgrid = "Pleasantness",
                             arousal_affectgrid      = "Arousal"))) +
  scale_fill_manual(values = c("ASMR trials"    = "forestgreen",
                               "No-ASMR trials" = "firebrick")) +
  labs(x = "Sound (1–13)", y = "Rating (±250)") +
  theme_bw(base_size = 12) +
  theme(legend.position = "none")

plot_split


## Pleasantness violins by sound
ggplot(df_final, aes(x = factor(sound), 
                     y = pleasantness_affectgrid)) +
  geom_violin(fill = "skyblue", trim = FALSE) +
  geom_boxplot(width = 0.1, outlier.size = 0.5) +   # median & quartiles
  labs(x = "Sound", y = "Pleasantness rating",
       title = "Distribution of pleasantness ratings by sound") +
  theme_minimal()

## Arousal violins by sound
ggplot(df_final, aes(x = factor(sound), 
                     y = arousal_affectgrid)) +
  geom_violin(fill = "salmon", trim = FALSE) +
  geom_boxplot(width = 0.1, outlier.size = 0.5) +
  labs(x = "Sound", y = "Arousal rating",
       title = "Distribution of arousal ratings by sound") +
  theme_minimal()

# plain correlation
r_val <- cor(df_means$mean_pleasant,
             df_means$mean_arousal,
             use = "complete.obs",
             method = "pearson")
print(r_val)
# e.g. 0.66  (hypothetical)

# correlation + significance test
cor_test <- cor.test(df_means$mean_pleasant,
                     df_means$mean_arousal,
                     method = "pearson")

cor_test 

#violin plots with coloured scatter for asmr v non asmr coded

install.packages("ggbeeswarm")   # run once
library(ggbeeswarm)

library(ggplot2)
library(ggbeeswarm)   # nicer jitter than geom_jitter

#  Make sure asmr_experience is a factor with readable labels
df_final <- df_final %>%
  mutate(asmr_cat = factor(asmr_experience,
                           levels = c(0, 1),
                           labels = c("No-ASMR", "ASMR")))

ggplot(df_final,
       aes(x = factor(sound),
           y = pleasantness_affectgrid,
           fill = asmr_cat,           # violin fill by ASMR?
           colour = asmr_cat)) +      # point colour by ASMR
  geom_violin(alpha = 0.25, trim = FALSE, colour = NA) +   # semi-transparent shell
  geom_beeswarm(dodge.width = 0.6, cex = 1.2, size = 0.5) +            # individual dots
  scale_fill_manual(values = c("grey80", "skyblue")) +
  scale_colour_manual(values = c("grey30", "royalblue")) +
  labs(x = "Sound (1-13)",
       y = "Pleasantness rating",
       fill = "Reported ASMR",
       colour = "Reported ASMR",
       title = "Pleasantness distribution per sound\n(coloured by ASMR-Yes/No)") +
  theme_minimal() +
  theme(legend.position = "top")
#13 scatter plots visualisation instead 
library(ggplot2)
library(dplyr)

## make a labelled 3-level factor for colour mapping
df_final <- df_final %>%
  mutate(asmr_cat = factor(asmr_experience,
                           levels = c(1, 0, NA),
                           labels = c("ASMR", "No-ASMR", "Unsure/NA")))

## faceted scatter: pleasantness  (x)  vs  arousal  (y)
ggplot(df_final,
       aes(x = pleasantness_affectgrid,
           y = arousal_affectgrid,
           colour = asmr_cat)) +
  geom_point(size = 2, alpha = 0.8) +
  facet_wrap(~ sound, nrow = 3) +         # 13 panels (sounds 1-13)
  scale_colour_manual(values = c("ASMR"       = "forestgreen",
                                 "No-ASMR"    = "firebrick",
                                 "Unsure/NA"  = "darkorange")) +
  labs(
    x = "Pleasantness rating",
    y = "Arousal rating",
    colour = "Participant response",
    title = "Per-sound scatter of participant ratings\n(colour = ASMR report)"
  ) +
  theme_minimal() +
  theme(legend.position = "top")
#all violin plots for asmr and non asmr and pleasant and arousal in 4 plots
library(ggplot2)
library(dplyr)

# ensure ASMR column is numeric 1 / 0 / NA
# df_final already has: asmr_experience (1 = Yes, 0 = No, NA = unsure)

# A helper function to make one violin figure -----------------------------
make_violin <- function(data, rating_col, title_text, fill_col) {
  ggplot(data,
         aes(x = factor(sound),
             y = .data[[rating_col]])) +
    geom_violin(fill = fill_col, trim = FALSE, colour = NA, alpha = .4) +
    geom_boxplot(width = 0.1, outlier.shape = NA, colour = "grey20") +
    labs(
      x = "Sound (1-13)",
      y = rating_col,
      title = title_text
    ) +
    theme_minimal()
}

make_violin <- function(df, yvar, title, fill_col) {
  ggplot(df, aes(x = factor(sound), y = .data[[yvar]])) +
    geom_violin(scale = "width",          # ‼ key change
                trim  = TRUE,
                fill  = fill_col,
                alpha = .6) +
    geom_boxplot(width = .1, fill = "white", outlier.shape = NA) +
    labs(title = title,
         x = "Sound (1–13)",
         y = yvar) +
    theme_bw(base_size = 12)
}
# ----------------- 1) Pleasantness • ASMR trials -------------------------
plot_pleasant_ASMR <- df_final %>% 
  filter(asmr_experience == 1) %>% 
  make_violin("pleasantness_affectgrid",
              "Pleasantness ratings • ASMR trials only",
              fill_col = "forestgreen")

# ----------------- 2) Pleasantness • No-ASMR trials ----------------------
plot_pleasant_No <- df_final %>% 
  filter(asmr_experience == 0) %>% 
  make_violin("pleasantness_affectgrid",
              "Pleasantness ratings • No-ASMR trials only",
              fill_col = "firebrick")

# ----------------- 3) Arousal • ASMR trials ------------------------------
plot_arousal_ASMR <- df_final %>% 
  filter(asmr_experience == 1) %>% 
  make_violin("arousal_affectgrid",
              "Arousal ratings • ASMR trials only",
              fill_col = "forestgreen")

# ----------------- 4) Arousal • No-ASMR trials ---------------------------
plot_arousal_No <- df_final %>% 
  filter(asmr_experience == 0) %>% 
  make_violin("arousal_affectgrid",
              "Arousal ratings • No-ASMR trials only",
              fill_col = "firebrick")

# ---- display or save ----------------------------------------------------
plot_pleasant_ASMR
plot_pleasant_No
plot_arousal_ASMR
plot_arousal_No
#side by side
install.packages("patchwork")
library(patchwork)   #  
(plot_pleasant_ASMR | plot_pleasant_No) /
  (plot_arousal_ASMR  | plot_arousal_No)
#stats for LMM ASMR vs non ASMR 
install.packages("lmerTest")
install.packages("effectsize")

############################################################
#  1)  Load packages each session
############################################################
library(lmerTest)      # lmer() + p-values
library(effectsize)    # standardize_parameters()

############################################################
#  2)  Fit the two mixed models
############################################################
m_arousal <- lmer(
  arousal_affectgrid ~ asmr_experience +
    (1 | participant) + (1 | sound),
  data = df_final)

m_pleasant <- lmer(
  pleasantness_affectgrid ~ asmr_experience +
    (1 | participant) + (1 | sound),
  data = df_final)

############################################################
#  3)  Cohen’s d (effect size) for the ASMR fixed effect
############################################################
standardize_parameters(m_arousal,  effectsize_type = "d")
standardize_parameters(m_pleasant, effectsize_type = "d")



#Quick confirmatory test – Participant-level paired t-test
#Collapse each participant to two means (ASMR-Yes vs No), 
#then test the paired difference.

df_pair <- df_final %>% 
  group_by(participant, asmr_experience) %>% 
  summarise(across(c(pleasantness_affectgrid, arousal_affectgrid),
                   mean, na.rm = TRUE), .groups = "drop") %>% 
  pivot_wider(names_from = asmr_experience,
              values_from = c(pleasantness_affectgrid, arousal_affectgrid),
              names_glue = "{.value}_{asmr_experience}")  # columns …_0 and …_1

t.test(df_pair$pleasantness_affectgrid_1,
       df_pair$pleasantness_affectgrid_0, paired = TRUE)

t.test(df_pair$arousal_affectgrid_1,
       df_pair$arousal_affectgrid_0, paired = TRUE)

#leven's test comparing spread 
install.packages("car")
car::leveneTest(pleasantness_affectgrid ~ asmr_experience, data = df_final)
car::leveneTest(arousal_affectgrid      ~ asmr_experience, data = df_final)

#predicting if they got ASMR from the ratings of pleasantness and arousal 
library(lme4)
library(lmerTest)

# Ensure predictors are numeric & scaled (improves convergence/readability)
df_final <- df_final %>% 
  mutate(
    p_z = scale(pleasantness_affectgrid),
    a_z = scale(arousal_affectgrid)
  )

m_logit <- glmer(
  asmr_experience ~ p_z + a_z +
    (1 | participant) + (1 | sound),
  data = df_final,
  family = binomial
)

summary(m_logit)
#likelihood ratio test: Shows whether adding pleasantness+arousal improves fit beyond chance


#for ASMR trials, do pleasantness and arousal predict intensity ratings? 
library(dplyr)
library(lmerTest)

# 1)  Clean the intensity column once -------------------------------
df_final <- df_final %>% 
  mutate(intensity = readr::parse_number(intensity))   # "N/A" → NA, "45" → 45

# 2)  Keep only ASMR trials with a valid intensity score ------------
df_intensity <- df_final %>% 
  filter(asmr_experience == 1,          # ASMR-Yes trials
         !is.na(intensity))             # numeric rating present

# Optional: scale predictors (keeps coefficients comparable)
df_intensity <- df_intensity %>% 
  mutate(
    p_z = scale(pleasantness_affectgrid),
    a_z = scale(arousal_affectgrid)
  )

# 3)  Fit the mixed-effects model -----------------------------------
m_intensity <- lmer(
  intensity ~ p_z + a_z +
    (1 | participant) + (1 | sound),
  data = df_intensity)

summary(m_intensity)

#cleveland pot of variance spread for ASMR v nonASMR in each dimension
library(dplyr)
library(tidyr)
library(ggplot2)

# ------------------------------------------------------------------
# 1. Keep only Yes (1) and No (0) trials ---------------------------
# ------------------------------------------------------------------
df_yesno <- df_final %>% 
  filter(asmr_experience %in% c(0, 1))

# ------------------------------------------------------------------
# 2. Compute SD per sound × condition ------------------------------
# ------------------------------------------------------------------
sd_by_sound <- df_yesno %>% 
  group_by(sound, asmr_experience) %>% 
  summarise(
    pleas_sd = sd(pleasantness_affectgrid, na.rm = TRUE),
    arous_sd = sd(arousal_affectgrid,      na.rm = TRUE),
    .groups  = "drop"
  ) %>% 
  mutate(
    asmr_label = ifelse(asmr_experience == 1, "ASMR", "No-ASMR")
  )

# ------------------------------------------------------------------
# 3. Pivot to long & keep only paired clips ------------------------
# ------------------------------------------------------------------
sd_long <- sd_by_sound %>% 
  pivot_longer(
    cols      = c(pleas_sd, arous_sd),
    names_to  = "dimension",
    values_to = "sd"
  ) %>% 
  group_by(sound, dimension) %>% 
  filter(n() == 2) %>%        # retain only clips that have BOTH SDs
  ungroup()

# ------------------------------------------------------------------
# 4. Cleveland plot with dodged points -----------------------------
# ------------------------------------------------------------------
ggplot(
  sd_long,
  aes(
    x     = sd,
    y     = factor(sound),
    colour = asmr_label,
    group  = interaction(sound, dimension)
  )
) +
  geom_line(linewidth = 0.5) +                          # connecting segment
  geom_point(                                           # ↓ dodge shows both dots
    position = position_dodge(width = 0.6),             # 0.6 ≈ shift ±0.3 on y
    size     = 2
    # ,
    # shape    = 19                # uncomment and tweak if you also add shapes
  ) +
  facet_wrap(
    ~ dimension,
    labeller = labeller(
      dimension = c(
        pleas_sd = "Pleasantness",
        arous_sd = "Arousal"
      )
    )
  ) +
  scale_colour_manual(
    values = c("ASMR" = "forestgreen",
               "No-ASMR" = "firebrick")
  ) +
  labs(
    title = "Spread of ratings by sound and condition",
    x     = "Within-sound SD (±250 scale)",
    y     = "Sound clip"
  ) +
  theme_bw(base_size = 12) +
  theme(
    legend.title = element_blank()
  )

# ------------------------------------------------------------------
# Optional: give different shapes so overlap is visible even without dodge
# ------------------------------------------------------------------
# + scale_shape_manual(values = c("ASMR" = 16, "No-ASMR" = 17)) inside aes(...)



library(dplyr)
library(tidyr)
library(ggplot2)

# 1. keep only Yes (1) and No (0) trials
df_yesno <- df_final %>% 
  filter(asmr_experience %in% c(0, 1))

# 2. compute SD per sound × condition
sd_by_sound <- df_yesno %>% 
  group_by(sound, asmr_experience) %>% 
  summarise(pleas_sd = sd(pleasantness_affectgrid),
            arous_sd = sd(arousal_affectgrid),
            .groups  = "drop") %>% 
  mutate(asmr_label = ifelse(asmr_experience == 1, "ASMR", "No-ASMR"))

# 3. pivot to long and keep only clips that have both ASMR and No-ASMR SDs
sd_long <- sd_by_sound %>% 
  pivot_longer(cols      = c(pleas_sd, arous_sd),
               names_to  = "dimension",
               values_to = "sd") %>% 
  group_by(sound, dimension) %>% 
  filter(n() == 2) %>%      # retain paired SDs only
  ungroup()

# 4. Cleveland plot
ggplot(sd_long,
       aes(x = sd, y = factor(sound),
           colour = asmr_label,
           group  = interaction(sound, dimension))) +
  geom_line(linewidth = 0.5) +
  geom_point(size = 2) +
  facet_wrap(~ dimension,
             labeller = labeller(
               dimension = c(pleas_sd = "Pleasantness",
                             arous_sd = "Arousal"))) +
  scale_colour_manual(values = c("ASMR" = "forestgreen",
                                 "No-ASMR" = "firebrick")) +
  labs(title = "Spread of ratings by sound and condition",
       x      = "Within-sound SD (±250 scale)",
       y      = "Sound clip") +
  theme_bw(base_size = 12) +
  theme(legend.title = element_blank())

#raw data check
df_yesno %>%                     # <— data after removing “Not-sure”
  filter(sound %in% c(4, 5, 7, 8, 12, 13)) %>%   # clips with missing reds
  group_by(sound, asmr_experience) %>% 
  summarise(n_trials = n(),
            sd_pleasant = sd(pleasantness_affectgrid, na.rm = TRUE),
            sd_arousal  = sd(arousal_affectgrid,      na.rm = TRUE),
            .groups = "drop")
#dose-repsonse plot for pleasantness and intensity of tingles
library(dplyr)
library(ggplot2)

library(tidyverse)
library(lme4)
library(lmerTest)
library(broom.mixed)

# -------------------------------------------
# 1. Filter to ASMR-YES trials with intensity
# -------------------------------------------
df_intensity <- df_final %>% 
  filter(asmr_experience == 1,
         !is.na(intensity))

# -------------------------------------------
# 2. Plot: dose-response with colour by clip
# -------------------------------------------
ggplot(df_intensity,
       aes(x = pleasantness_affectgrid,
           y = intensity,
           colour = factor(sound))) +        ### ← added
  geom_point(alpha = .6, size = 1.8) +
  geom_smooth(method = "lm",
              se      = TRUE,
              colour  = "black",             ### ← edited (single line on top)
              fill    = "grey80",
              linewidth = .9) +
  labs(title    = "Scatter plot of the relationship between pleasantness (z-scored) and tingle intensity",
       subtitle = "Points coloured by sound clip; black line = overall fixed-effect slope (β ≈ 0.90)",
       x        = "Pleasantness rating (±250 grid)",
       y        = "Tingle intensity (0–100)",
       colour   = "Sound\nclip") +           ### ← legend title
  theme_bw(base_size = 12) +
  theme(legend.position = "right")
#different neater scatterplot of tingles and pleasure 
library(dplyr)
library(ggplot2)

df_intensity <- df_final %>% 
  filter(asmr_experience == 1,
         !is.na(intensity))

p <- ggplot(df_intensity,
            aes(x = pleasantness_affectgrid,
                y = intensity,
                colour = factor(sound))) +
  geom_point(alpha = .6, size = 1.8) +
  geom_smooth(method = "lm",
              se = TRUE,
              colour = "black",    # one overall regression line
              fill = "grey80",
              linewidth = .9) +
  labs(
    title  = "Scatter plot of the relationship between pleasantness (z-scored) and tingle intensity",
    x      = "Pleasantness rating (z-scored)",
    y      = "Tingle intensity (0–100)",
    colour = "Sound\nclip"
  ) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "right",
    plot.title = element_text(
      hjust = 0.5,     # centres the title
      size = 11,       # makes it slightly smaller
      face = "plain"   # optional: remove bold
    )
  )

# Save as TIFF at 300 dpi
ggsave(
  filename = "pleasantness_tingles.tiff",
  plot = p,
  dpi = 300,
  width = 18,    # cm (double-column width for Frontiers)
  height = 12,   # cm
  units = "cm"
)

#actual saveable scatterplot of pleasantness and tingles with z score 
library(dplyr)
library(ggplot2)

# 1. Filter and z-score
df_intensity <- df_final %>%
  filter(asmr_experience == 1,
         !is.na(intensity),
         !is.na(pleasantness_affectgrid)) %>%
  mutate(
    pleasantness_z = as.numeric(scale(pleasantness_affectgrid)),
    sound = factor(sound)
  )

# 2. Create the plot
p <- ggplot(df_intensity,
            aes(x = pleasantness_z,
                y = intensity,
                colour = sound)) +
  geom_point(alpha = 0.6, size = 1.8) +
  geom_smooth(method = "lm",
              se = TRUE,
              colour = "black",        # overall regression line
              fill = "grey80",
              linewidth = 0.9) +
  labs(
    title  = "Scatter plot of the relationship between pleasantness (z-scored) and tingle intensity",
    x      = "Pleasantness rating (z-scored)",
    y      = "Tingle intensity (0–100)",
    colour = "Sound\nclip"
  ) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "right",
    plot.title = element_text(hjust = 0.5, size = 11, face = "plain")
  )

# 3. Create output folder on Desktop
out_dir <- "~/Desktop/figures"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# 4. Save as TIFF at 300 dpi, double-column size
ggsave(
  filename = file.path(out_dir, "pleasantness_tingles.tiff"),
  plot = p,
  dpi = 300,
  width = 18,   # cm (double-column width for Frontiers)
  height = 12,  # cm
  units = "cm"
)

#newest tingle plot for frontiers, without regression lines and with y axis correct
library(dplyr)
library(ggplot2)

# 1. Filter and clean data
df_intensity <- df_final %>%
  filter(asmr_experience == 1,
         !is.na(intensity),
         !is.na(pleasantness_affectgrid)) %>%
  mutate(
    pleasantness_z = as.numeric(scale(pleasantness_affectgrid)),
    intensity = as.numeric(intensity),   # ensure proper numeric order on y-axis
    sound = factor(sound)
  )

# 2. Create scatter plot only (no regression lines)
p <- ggplot(df_intensity,
            aes(x = pleasantness_z,
                y = intensity,
                colour = sound)) +
  geom_point(alpha = 0.6, size = 1.8) +
  labs(
    title  = "Scatter plot of the relationship between pleasantness (z-scored) and tingle intensity",
    x      = "Pleasantness rating (z-scored)",
    y      = "Tingle intensity (0–100)",
    colour = "Sound\nclip"
  ) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "right",
    plot.title = element_text(hjust = 0.5, size = 11, face = "plain")
  )

# 3. Create output folder on Desktop
out_dir <- "~/Desktop/figures"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# 4. Save as TIFF at 300 dpi
ggsave(
  filename = file.path(out_dir, "pleasantness_tingles_scatter.tiff"),
  plot = p,
  dpi = 300,
  width = 18,   # cm (double-column width)
  height = 12,  # cm
  units = "cm"
)

#interaction tests 

install.packages(c("ggeffects", "patchwork", "viridis"))
library(lme4)        # mixed models
library(lmerTest)    # p-values for lmer
library(ggeffects)   # easy marginal-effects predictions
library(ggplot2)
library(patchwork)   # arrange plots
library(viridis)     # nice heat-map palette

############################################################
# 1)  Scale predictors once per data set
############################################################
df_final <- df_final %>% 
  mutate(
    p_z = scale(pleasantness_affectgrid,  center = TRUE, scale = TRUE)[,1],
    a_z = scale(arousal_affectgrid,       center = TRUE, scale = TRUE)[,1]
  )

###################################################
# tingle interaction to make a table for output with t values 

library(dplyr)
library(lme4)
library(lmerTest)     # adds df, t, p to lmer()
library(broom.mixed)

## 0) Inspect types (optional but helpful)
str(df_final)

## 1) Make sure predictors are z-scored once
df_final <- df_final %>%
  mutate(
    p_z = as.numeric(scale(pleasantness_affectgrid)),
    a_z = as.numeric(scale(arousal_affectgrid))
  )

## 2) Make sure intensity is numeric (handle factor/character safely)
df_final <- df_final %>%
  mutate(
    intensity = suppressWarnings(as.numeric(as.character(intensity)))
  )

## Check for NAs introduced by coercion (optional)
sum(is.na(df_final$intensity))

## 3) Filter ASMR-positive trials with valid intensity
df_intensity <- df_final %>%
  filter(asmr_experience == 1, !is.na(intensity))

## 4) Fit linear mixed models for tingle intensity
m_intensity <- lmer(
  intensity ~ p_z + a_z + (1|participant) + (1|sound),
  data = df_intensity
)

m_intensity_int <- lmer(
  intensity ~ p_z * a_z + (1|participant) + (1|sound),
  data = df_intensity
)

## 5) Summaries (includes β, SE, df, t, p)
summary(m_intensity)
summary(m_intensity_int)

## 6) Tidy table for the interaction model (publication-ready)
intI_tbl <- tidy(m_intensity_int, effects = "fixed", conf.int = TRUE) %>%
  mutate(
    term = dplyr::recode(term,
                         `(Intercept)` = "Intercept",
                         `p_z` = "Pleasantness (z)",
                         `a_z` = "Arousal (z)",
                         `p_z:a_z` = "Pleasantness × Arousal"
    )
  ) %>%
  select(term, estimate, std.error, df, statistic, p.value, conf.low, conf.high)

print(intI_tbl, n = Inf)

## 7) Optional: round and export to CSV
intI_tbl_round <- intI_tbl %>%
  mutate(
    estimate  = round(estimate, 2),
    std.error = round(std.error, 2),
    df        = round(df, 1),
    statistic = round(statistic, 2),
    p.value   = signif(p.value, 3),
    conf.low  = round(conf.low, 2),
    conf.high = round(conf.high, 2)
  )

write.csv(intI_tbl_round, "tingle_intensity_interaction_fixed_effects.csv", row.names = FALSE)





############################################################
# 2)  Logistic model: Does Arousal *moderate* the Pleasantness
#     effect on the ASMR (Yes/No) decision?
############################################################
m_logit_int <- glmer(
  asmr_experience ~ p_z * a_z +           # interaction term
    (1 | participant) + (1 | sound),
  data   = df_final,
  family = binomial
)

library(broom.mixed)

logit_table <- tidy(m_logit_int, effects = "fixed", conf.int = TRUE)
print(logit_table)

summary(m_logit_int)          # check interaction p-value
anova(m_logit, m_logit_int)   # LR test vs. additive model

############################################################
# 3)  Mixed model: Does the interaction influence
#     *tingle intensity* on ASMR-positive trials?
############################################################
df_intensity <- df_final %>% 
  filter(asmr_experience == 1,
         !is.na(intensity)) %>% 
  mutate(a_z = scale(arousal_affectgrid)[,1],
         p_z = scale(pleasantness_affectgrid)[,1])

tidy(m_intensity_int, effects = "fixed", conf.int = TRUE)


############################################################
# 4)  Visualisation A:  Heat-map of predicted ASMR probability
############################################################
# 4a.  Generate a grid over ±2 SD in each dimension
newdat <- expand.grid(
  p_z = seq(-2, 2, length = 51),
  a_z = seq(-2, 2, length = 51)
)

# 4b.  Predicted probabilities
newdat$pred <- predict(m_logit_int, newdata = newdat,
                       re.form = NA, type = "response")

# 4c.  Heat-map
heat_ASMR <- ggplot(newdat,
                    aes(x = p_z, y = a_z, fill = pred)) +
  geom_tile() +
  coord_equal() +
  scale_fill_viridis(name = "P(ASMR)",
                     option = "B", limits = c(0,1)) +
  labs(x = "Pleasantness (z)", y = "Arousal (z)",
       title = "Predicted ASMR probability\nPleasantness × Arousal") +
  theme_minimal(base_size = 12)

############################################################
# 5)  Visualisation B:  Interaction lines for intensity
############################################################
# 5a.  Predicted intensity at low / mid / high arousal
pred_int <- ggpredict(m_intensity_int,
                      terms = c("p_z [pretty]",  # x-axis
                                "a_z [-1,0,1]"))  #  −1σ, mean, +1σ

# 5b.  Plot
lines_int <- ggplot(pred_int,
                    aes(x = x, y = predicted,
                        colour = group)) +
  geom_line(size = 1.2) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high,
                  fill = group), alpha = .2, colour = NA) +
  scale_colour_manual(values = c("#d1495b", "#00798c", "#edae49"),
                      labels  = c("-1 σ arousal",
                                  "mean arousal",
                                  "+1 σ arousal")) +
  scale_fill_manual(values = c("#d1495b", "#00798c", "#edae49"), guide = "none") +
  labs(x = "Pleasantness (z)",
       y = "Predicted tingle intensity",
       colour = "Arousal level",
       title = "Interaction effect on tingle intensity") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top")

############################################################
# 6)  Show plots side-by-side  -----------------------------
############################################################
heat_ASMR | lines_int
ggsave("section4_heatmap.tiff", heat_ASMR, dpi = 300, width = 7, height = 6, units = "in")
ggsave("section4_intensityLines.tiff", lines_int, dpi = 300, width = 7, height = 6, units = "in")

ggsave(
  filename = "section4_ASMR_prob_heatmap.png",
  plot     = heat_ASMR,
  width    = 120,
  height   = 120,
  units    = "mm",
  dpi      = 300,
  bg       = "white"         # ensures white background
)

#get the fixed effects, random effects, confidence intervals, model fit
install.packages("broom.mixed")
install.packages("performance")
install.packages("sjPlot")  
library(performance)
library(broom.mixed)
library(effectsize)

# Extract summaries
tidy(m_pleasant, conf.int = TRUE, effects = "fixed")
tidy(m_arousal, conf.int = TRUE, effects = "fixed")

# Random effects
VarCorr(m_pleasant)
VarCorr(m_arousal)

# R-squared
r2(m_pleasant)
r2(m_arousal)

# Cohen's d
standardize_parameters(m_pleasant, effectsize_type = "d")
standardize_parameters(m_arousal, effectsize_type = "d")


library(broom.mixed)
tidy(m_logit_int, effects = "fixed", conf.int = TRUE)


#reviewer additions: 
# Install once if needed, this is for the logistical table
install.packages(c("broom.mixed","performance","MuMIn","dplyr"))

library(broom.mixed)
library(performance)
library(MuMIn)
library(dplyr)

## Fixed effects table with ORs
tab_logit <- tidy(m_logit_int, effects = "fixed", conf.int = TRUE) %>%
  mutate(
    OR      = exp(estimate),
    OR_low  = exp(conf.low),
    OR_high = exp(conf.high)
  ) %>%
  transmute(
    Term = dplyr::recode(term,
                         `(Intercept)` = "Intercept",
                         `p_z`         = "Pleasantness (z)",
                         `a_z`         = "Arousal (z)",
                         `p_z:a_z`     = "Pleasantness × Arousal"),
    Estimate = estimate,
    SE = std.error,
    Z  = statistic,
    p  = p.value,
    `95% CI (β low)` = conf.low,
    `95% CI (β high)`= conf.high,
    OR = OR,
    `95% CI (OR low)`= OR_low,
    `95% CI (OR high)`= OR_high
  )

# Round for manuscript
tab_logit_print <- tab_logit %>%
  mutate(across(where(is.numeric), ~signif(., 3)))

tab_logit_print
# write.csv(tab_logit_print, "Table1_logistic_with_OR.csv", row.names = FALSE)

## Model fit / R²
perf_logit <- model_performance(m_logit_int)     # AIC, BIC, logLik, RMSE (for GLMM), etc.
perf_logit

r2_logit <- r2(m_logit_int)                      # Marginal & Conditional R²
r2_logit

# (Optionally also report MuMIn GLMM R²)
r.squaredGLMM(m_logit_int)
print(tab_logit_print, n=Inf, width=Inf)

#this is for the linear mixed model table 
library(broom.mixed)
library(performance)
library(r2glmm)
library(dplyr)

## Partial R² for each fixed effect (Nakagawa–Johnson–Schielzeth method)
partR <- r2beta(m_intensity_int, method = "nsj", partial = TRUE)
# partR has columns: Effect, Rsq, upper.CL, lower.CL

## Fixed effects table + merge partial R² (note the CL column names)
tab_lmm <- tidy(m_intensity_int, effects = "fixed", conf.int = TRUE) %>%
  transmute(
    Term = dplyr::recode(term,
                         `(Intercept)` = "Intercept",
                         `p_z`         = "Pleasantness (z)",
                         `a_z`         = "Arousal (z)",
                         `p_z:a_z`     = "Pleasantness × Arousal"
    ),
    Beta = estimate,
    SE   = std.error,
    df   = df,
    t    = statistic,
    p    = p.value,
    `95% CI (low)`  = conf.low,
    `95% CI (high)` = conf.high
  ) %>%
  left_join(
    partR %>%
      transmute(
        Term = dplyr::recode(Effect,
                             `(Intercept)` = "Intercept",
                             `p_z`         = "Pleasantness (z)",
                             `a_z`         = "Arousal (z)",
                             `p_z:a_z`     = "Pleasantness × Arousal"
        ),
        `Partial R²`         = Rsq,
        `Partial R² CI low`  = lower.CL,
        `Partial R² CI high` = upper.CL
      ),
    by = "Term"
  )

## Round for manuscript
tab_lmm_print <- tab_lmm %>%
  mutate(across(where(is.numeric), ~signif(., 3)))

tab_lmm_print
#fully standardise 
library(lmerTest)

m_intensity_int_std <- lmer(
  scale(intensity) ~ p_z * a_z + (1|participant) + (1|sound),
  data = df_intensity
)

tab_lmm_std <- tidy(m_intensity_int_std, effects = "fixed", conf.int = TRUE) %>%
  transmute(
    Term = dplyr::recode(term,
                         `(Intercept)` = "Intercept",
                         `p_z`         = "Pleasantness (z)",
                         `a_z`         = "Arousal (z)",
                         `p_z:a_z`     = "Pleasantness × Arousal"
    ),
    `Std. Beta (β*)` = estimate,
    SE   = std.error,
    df   = df,
    t    = statistic,
    p    = p.value,
    `95% CI (low)`  = conf.low,
    `95% CI (high)` = conf.high
  )

tab_lmm_std
#model fit indices
perf_lmm <- model_performance(m_intensity_int)
perf_lmm
# AIC = 589.585, BIC = 609.974, R2(marginal) = 0.146, R2(conditional) = 0.529, ICC = 0.448

r2(m_intensity_int)
# same R² values

r.squaredGLMM(m_intensity_int)
# MuMIn corroboration (R2m ≈ .146, R2c ≈ .529)

