# Notebook Rendering

Total .qmd notebooks in `project/docs/`: **96**

**FROZEN** — executed locally with Quarto 1.4.557, outputs cached in `project/_freeze/` and rendered from cache at build time
**REGULAR** — no freeze cache; renders live (or as static code) in the Docker build

Five FROZEN-set notebooks have no executable code cells and therefore no freeze cache — they render as plain markdown each build: `econ490-r/01_Setting_Up`, `econ490-stata/01_Setting_Up`, `econ490-pystata/01_Setting_Up_PyStata`, `intro_to_python_network_analysis`, `advanced_vocalization_draft`. (82 FROZEN = 77 cached + these 5.)

## 1. Getting Started (5 notebooks)

- `getting_started_intro_to_python` — **FROZEN**
- `getting_started_intro_to_data2` — **FROZEN**
- `getting_started_intro_to_jupyter` — **FROZEN**
- `getting_started_intro_to_r` — **FROZEN**
- `getting_started_intro_to_data1` — REGULAR

## 2. Beginner (11 notebooks)

- `beginner_central_tendency` — **FROZEN**
- `beginner_confidence_intervals` — **FROZEN**
- `beginner_dispersion_and_dependence` — **FROZEN**
- `beginner_distributions` — **FROZEN**
- `beginner_hypothesis_testing` — **FROZEN**
- `beginner_intro_to_central_tendency` — **FROZEN**
- `beginner_intro_to_data_visualization1` — **FROZEN**
- `beginner_intro_to_data_visualization2` — **FROZEN**
- `beginner_intro_to_statistics2` — **FROZEN**
- `beginner_sampling_distributions` — **FROZEN**
- `beginner_intro_to_statistics1` — REGULAR

## 3. Intermediate (5 notebooks)

- `GTsummary` — **FROZEN**
- `intermediate_interactions_and_nonlinear_terms` — **FROZEN**
- `intermediate_intro_to_regression` — **FROZEN**
- `intermediate_multiple_regression` — **FROZEN**
- `intermediate_issues_in_regression` — REGULAR

## 4. Advanced (19 notebooks)

- `advanced_instrumental_variables1` — **FROZEN**
- `advanced_instrumental_variables2` — **FROZEN**
- `advanced_linear_differencing` — **FROZEN**
- `advanced_synthetic_control` — **FROZEN**
- `advanced_vocalization_draft` — **FROZEN**
- `advanced_word_embeddings_python_version` — **FROZEN**
- `intro_to_python_network_analysis` — **FROZEN**
- `network_analysis_notebook` — **FROZEN**
- `network_analysis_notebook_II` — **FROZEN**
- `advanced_geospatial` — **FROZEN**
- `advanced_geospatial_2` — **FROZEN**
- `advanced_classification_and_clustering` — REGULAR
- `advanced_difference_in_differences` — REGULAR
- `advanced_word_embeddings_r_version` — REGULAR
- `advanced_panel_data` — REGULAR
- `advanced_llm_apis2` — REGULAR
- `fine_tuning_llm` — REGULAR
- `sentiment_analysis` — REGULAR
- `advanced_transcription_whisper` — REGULAR

## 5. econ490-r (17 notebooks)

- 01_Setting_Up — **FROZEN**
- 02_Working_Rscripts — **FROZEN**
- 03_R_Essentials — **FROZEN**
- 04_Opening_Data_Sets — **FROZEN**
- 05_Creating_Variables — **FROZEN**
- 07_Combining_Datasets — **FROZEN**
- 08_ggplot_graphs — **FROZEN**
- 09_Combining_Graphs — **FROZEN**
- 10_Linear_Reg — **FROZEN**
- 11_Exporting_Output — **FROZEN**
- 12_Dummy — **FROZEN**
- 13_PostReg — **FROZEN**
- 14_Panel_Data — **FROZEN**
- 15_Diff_in_Diff — **FROZEN**
- 16_IV — **FROZEN**
- 06_Within_Group — REGULAR
- 17_Wf_Guide — REGULAR

## 6. econ490-stata (18 notebooks)

- 01 through 18 — **ALL FROZEN**

## 7. econ490-pystata (18 notebooks)

- 01 through 18 — **ALL FROZEN**

## 8. Projects (3 notebooks)

- `Projects_Example_Project_ECON325` — **FROZEN**
- `Projects_Example_Project_ECON326` — **FROZEN**
- `intermediate_stargazer` — REGULAR

## Overall Totals

- **FROZEN**: 82
- **REGULAR**: 14
- **Total**: 96

## Excluded Notebooks (4)

These were not frozen due to hardware and maintanance reasons:

- `advanced_llm_apis2` — Requires a locally running Ollama server.
- `fine_tuning_llm` — Requires GPU compute and large model downloads.
- `sentiment_analysis` — Reddit blocks unauthenticated API scraping.
- `advanced_transcription_whisper` — Requires ffmpeg and a HuggingFace auth token.
