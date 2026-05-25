# Taken from COMET will have to be adapted for Praxis
# Build stage 2025-06-04
FROM jlgraves/comet-test:test AS builder

WORKDIR /app

COPY ./meta/building/renv.lock ./project ./

#Removes the rendered .qmd's 
RUN rm -f ./docs/SOCI-415/soci_415_network_analysis.qmd
RUN rm -f ./docs/SOCI-415/kinmatrix.qmd
RUN rm -f ./docs/SOCI-415/cbdb_dataset.qmd
RUN rm -f ./docs/ECON-227/llm_distributions.qmd
RUN rm -f ./docs/hist_workshop/text_embeddings_workshop.qmd
RUN rm -f ./docs/intro_to_cnns/intro_to_cnn.qmd
RUN rm -f ./docs/intro_to_convolutions/intro_to_convolution.qmd
RUN rm -f ./docs/intro_to_deep_learning/intro_to_fundamental_ML.qmd
RUN rm -f ./docs/AMNE-376/amne_376_image_embedding.qmd
RUN rm -f ./docs/SOCI-280/soci_280_bert.qmd
RUN rm -f ./docs/OCR/ocr_notebook.qmd
RUN rm -f ./docs/image_analysis/image_analysis.qmd
RUN rm -f ./docs/CTree_CEA/political_economy.qmd
RUN rm -f ./docs/CTree_CEA/llm_distributions.qmd
RUN rm -f ./docs/1_Getting_Started/getting_started_intro_to_data/getting_started_intro_to_data2.qmd
RUN rm -f ./docs/1_Getting_Started/getting_started_intro_to_jupyter/getting_started_intro_to_jupyter.qmd
RUN rm -f ./docs/1_Getting_Started/getting_started_intro_to_python/getting_started_intro_to_python.qmd
RUN rm -f ./docs/1_Getting_Started/getting_started_intro_to_r/getting_started_intro_to_r.qmd
RUN rm -f ./docs/2_Beginner/beginner_central_tendency/beginner_central_tendency.qmd
RUN rm -f ./docs/2_Beginner/beginner_confidence_intervals/beginner_confidence_intervals.qmd
RUN rm -f ./docs/2_Beginner/beginner_dispersion_and_dependence/beginner_dispersion_and_dependence.qmd
RUN rm -f ./docs/2_Beginner/beginner_distributions/beginner_distributions.qmd
RUN rm -f ./docs/2_Beginner/beginner_hypothesis_testing/beginner_hypothesis_testing.qmd
RUN rm -f ./docs/2_Beginner/beginner_intro_to_central_tendency/beginner_intro_to_central_tendency.qmd
RUN rm -f ./docs/2_Beginner/beginner_intro_to_data_visualization1/beginner_intro_to_data_visualization1.qmd
RUN rm -f ./docs/2_Beginner/beginner_intro_to_data_visualization2/beginner_intro_to_data_visualization2.qmd
RUN rm -f ./docs/2_Beginner/beginner_intro_to_statistics2/beginner_intro_to_statistics2.qmd
RUN rm -f ./docs/2_Beginner/beginner_sampling_distributions/beginner_sampling_distributions.qmd
RUN rm -f ./docs/3_Intermediate/intermediate_interactions_and_nonlinear_terms/intermediate_interactions_and_nonlinear_terms.qmd
RUN rm -f ./docs/3_Intermediate/intermediate_intro_to_regression/intermediate_intro_to_regression.qmd
RUN rm -f ./docs/3_Intermediate/intermediate_multiple_regression/intermediate_multiple_regression.qmd
RUN rm -f ./docs/3_Intermediate/intermediate_summary_statistics/GTsummary.qmd
RUN rm -f ./docs/4_Advanced/advanced_geospatial/advanced_geospatial.qmd
RUN rm -f ./docs/4_Advanced/advanced_geospatial/advanced_geospatial_2.qmd
RUN rm -f ./docs/4_Advanced/advanced_instrumental_variables/advanced_instrumental_variables1.qmd
RUN rm -f ./docs/4_Advanced/advanced_instrumental_variables/advanced_instrumental_variables2.qmd
RUN rm -f ./docs/4_Advanced/advanced_linear_differencing/advanced_linear_differencing.qmd
RUN rm -f ./docs/4_Advanced/advanced_network_analysis/intro_to_python_network_analysis.qmd
RUN rm -f ./docs/4_Advanced/advanced_network_analysis/network_analysis_notebook.qmd
RUN rm -f ./docs/4_Advanced/advanced_network_analysis/network_analysis_notebook_II.qmd
RUN rm -f ./docs/4_Advanced/advanced_synthetic_control/advanced_synthetic_control.qmd
RUN rm -f ./docs/4_Advanced/advanced_vocalization/advanced_vocalization_draft.qmd
RUN rm -f ./docs/4_Advanced/advanced_word_embeddings/advanced_word_embeddings_python_version.qmd
RUN rm -f ./docs/5_Research/econ490-pystata/01_Setting_Up_PyStata.qmd
RUN rm -f ./docs/5_Research/econ490-pystata/02_Working_Dofiles.qmd
RUN rm -f ./docs/5_Research/econ490-pystata/09_Stata_Graphs.qmd
RUN rm -f ./docs/5_Research/econ490-pystata/11_Linear_Reg.qmd
RUN rm -f ./docs/5_Research/econ490-pystata/14_PostReg.qmd
RUN rm -f ./docs/5_Research/econ490-r/01_Setting_Up.qmd
RUN rm -f ./docs/5_Research/econ490-r/02_Working_Rscripts.qmd
RUN rm -f ./docs/5_Research/econ490-r/03_R_Essentials.qmd
RUN rm -f ./docs/5_Research/econ490-r/04_Opening_Data_Sets.qmd
RUN rm -f ./docs/5_Research/econ490-r/05_Creating_Variables.qmd
RUN rm -f ./docs/5_Research/econ490-r/07_Combining_Datasets.qmd
RUN rm -f ./docs/5_Research/econ490-r/08_ggplot_graphs.qmd
RUN rm -f ./docs/5_Research/econ490-r/09_Combining_Graphs.qmd
RUN rm -f ./docs/5_Research/econ490-r/10_Linear_Reg.qmd
RUN rm -f ./docs/5_Research/econ490-r/11_Exporting_Output.qmd
RUN rm -f ./docs/5_Research/econ490-r/12_Dummy.qmd
RUN rm -f ./docs/5_Research/econ490-r/13_PostReg.qmd
RUN rm -f ./docs/5_Research/econ490-r/14_Panel_Data.qmd
RUN rm -f ./docs/5_Research/econ490-r/15_Diff_in_Diff.qmd
RUN rm -f ./docs/5_Research/econ490-r/16_IV.qmd
RUN rm -f ./docs/5_Research/econ490-stata/01_Setting_Up.qmd
RUN rm -f ./docs/5_Research/econ490-stata/02_Working_Dofiles.qmd
RUN rm -f ./docs/5_Research/econ490-stata/03_Stata_Essentials.qmd
RUN rm -f ./docs/5_Research/econ490-stata/04_Locals_and_Globals.qmd
RUN rm -f ./docs/5_Research/econ490-stata/05_Opening_Data_Sets.qmd
RUN rm -f ./docs/5_Research/econ490-stata/06_Creating_Variables.qmd
RUN rm -f ./docs/5_Research/econ490-stata/07_Within_Group.qmd
RUN rm -f ./docs/5_Research/econ490-stata/08_Merge_Append.qmd
RUN rm -f ./docs/5_Research/econ490-stata/09_Stata_Graphs.qmd
RUN rm -f ./docs/5_Research/econ490-stata/10_Combining_Graphs.qmd
RUN rm -f ./docs/5_Research/econ490-stata/11_Linear_Reg.qmd
RUN rm -f ./docs/5_Research/econ490-stata/12_Exporting_Output.qmd
RUN rm -f ./docs/5_Research/econ490-stata/13_Dummy.qmd
RUN rm -f ./docs/5_Research/econ490-stata/14_PostReg.qmd
RUN rm -f ./docs/5_Research/econ490-stata/15_Panel_Data.qmd
RUN rm -f ./docs/5_Research/econ490-stata/16_Diff_in_Diff.qmd
RUN rm -f ./docs/5_Research/econ490-stata/17_IV.qmd
RUN rm -f ./docs/5_Research/econ490-stata/18_Wf_Guide2.qmd
RUN rm -f ./docs/6_Projects/projects_example_project_econ325/Projects_Example_Project_ECON325.qmd
RUN rm -f ./docs/6_Projects/projects_example_project_econ326/Projects_Example_Project_ECON326.qmd

RUN mkdir output

# Quarto render all documents + stub
RUN quarto render --output-dir /app/output

# Copy pre-rendered HTML file
COPY ./project/docs/SOCI-415/soci_415_network_analysis.html /app/output/docs/SOCI-415/
COPY ./project/docs/SOCI-415/kinmatrix.html /app/output/docs/SOCI-415/
COPY ./project/docs/SOCI-415/cbdb_dataset.html /app/output/docs/SOCI-415/
COPY ./project/docs/ECON-227/llm_distributions.html /app/output/docs/ECON-227/
COPY ./project/docs/hist_workshop/text_embeddings_workshop.html /app/output/docs/hist_workshop/
COPY ./project/docs/intro_to_cnns/intro_to_cnn.html /app/output/docs/intro_to_cnns/
COPY ./project/docs/intro_to_convolutions/intro_to_convolution.html /app/output/docs/intro_to_convolutions/
COPY ./project/docs/intro_to_deep_learning/intro_to_fundamental_ML.html /app/output/docs/intro_to_deep_learning/
COPY ./project/docs/AMNE-376/amne_376_image_embedding.html /app/output/docs/AMNE-376/
COPY ./project/docs/SOCI-280/soci_280_bert.html /app/output/docs/SOCI-280/
COPY ./project/docs/OCR/ocr_notebook.html /app/output/docs/OCR/
COPY ./project/docs/image_analysis/image_analysis.html /app/output/docs/image_analysis/
COPY ./project/docs/CTree_CEA/political_economy.html /app/output/docs/CTree_CEA/
COPY ./project/docs/CTree_CEA/llm_distributions.html /app/output/docs/CTree_CEA/
COPY ./project/docs/1_Getting_Started/getting_started_intro_to_data/getting_started_intro_to_data2.html /app/output/docs/1_Getting_Started/getting_started_intro_to_data/
COPY ./project/docs/1_Getting_Started/getting_started_intro_to_jupyter/getting_started_intro_to_jupyter.html /app/output/docs/1_Getting_Started/getting_started_intro_to_jupyter/
COPY ./project/docs/1_Getting_Started/getting_started_intro_to_python/getting_started_intro_to_python.html /app/output/docs/1_Getting_Started/getting_started_intro_to_python/
COPY ./project/docs/1_Getting_Started/getting_started_intro_to_r/getting_started_intro_to_r.html /app/output/docs/1_Getting_Started/getting_started_intro_to_r/
COPY ./project/docs/2_Beginner/beginner_central_tendency/beginner_central_tendency.html /app/output/docs/2_Beginner/beginner_central_tendency/
COPY ./project/docs/2_Beginner/beginner_confidence_intervals/beginner_confidence_intervals.html /app/output/docs/2_Beginner/beginner_confidence_intervals/
COPY ./project/docs/2_Beginner/beginner_dispersion_and_dependence/beginner_dispersion_and_dependence.html /app/output/docs/2_Beginner/beginner_dispersion_and_dependence/
COPY ./project/docs/2_Beginner/beginner_distributions/beginner_distributions.html /app/output/docs/2_Beginner/beginner_distributions/
COPY ./project/docs/2_Beginner/beginner_hypothesis_testing/beginner_hypothesis_testing.html /app/output/docs/2_Beginner/beginner_hypothesis_testing/
COPY ./project/docs/2_Beginner/beginner_intro_to_central_tendency/beginner_intro_to_central_tendency.html /app/output/docs/2_Beginner/beginner_intro_to_central_tendency/
COPY ./project/docs/2_Beginner/beginner_intro_to_data_visualization1/beginner_intro_to_data_visualization1.html /app/output/docs/2_Beginner/beginner_intro_to_data_visualization1/
COPY ./project/docs/2_Beginner/beginner_intro_to_data_visualization2/beginner_intro_to_data_visualization2.html /app/output/docs/2_Beginner/beginner_intro_to_data_visualization2/
COPY ./project/docs/2_Beginner/beginner_intro_to_statistics2/beginner_intro_to_statistics2.html /app/output/docs/2_Beginner/beginner_intro_to_statistics2/
COPY ./project/docs/2_Beginner/beginner_sampling_distributions/beginner_sampling_distributions.html /app/output/docs/2_Beginner/beginner_sampling_distributions/
COPY ./project/docs/3_Intermediate/intermediate_interactions_and_nonlinear_terms/intermediate_interactions_and_nonlinear_terms.html /app/output/docs/3_Intermediate/intermediate_interactions_and_nonlinear_terms/
COPY ./project/docs/3_Intermediate/intermediate_intro_to_regression/intermediate_intro_to_regression.html /app/output/docs/3_Intermediate/intermediate_intro_to_regression/
COPY ./project/docs/3_Intermediate/intermediate_multiple_regression/intermediate_multiple_regression.html /app/output/docs/3_Intermediate/intermediate_multiple_regression/
COPY ./project/docs/3_Intermediate/intermediate_summary_statistics/GTsummary.html /app/output/docs/3_Intermediate/intermediate_summary_statistics/
COPY ./project/docs/4_Advanced/advanced_geospatial/advanced_geospatial.html /app/output/docs/4_Advanced/advanced_geospatial/
COPY ./project/docs/4_Advanced/advanced_geospatial/advanced_geospatial_2.html /app/output/docs/4_Advanced/advanced_geospatial/
COPY ./project/docs/4_Advanced/advanced_instrumental_variables/advanced_instrumental_variables1.html /app/output/docs/4_Advanced/advanced_instrumental_variables/
COPY ./project/docs/4_Advanced/advanced_instrumental_variables/advanced_instrumental_variables2.html /app/output/docs/4_Advanced/advanced_instrumental_variables/
COPY ./project/docs/4_Advanced/advanced_linear_differencing/advanced_linear_differencing.html /app/output/docs/4_Advanced/advanced_linear_differencing/
COPY ./project/docs/4_Advanced/advanced_network_analysis/intro_to_python_network_analysis.html /app/output/docs/4_Advanced/advanced_network_analysis/
COPY ./project/docs/4_Advanced/advanced_network_analysis/network_analysis_notebook.html /app/output/docs/4_Advanced/advanced_network_analysis/
COPY ./project/docs/4_Advanced/advanced_network_analysis/network_analysis_notebook_II.html /app/output/docs/4_Advanced/advanced_network_analysis/
COPY ./project/docs/4_Advanced/advanced_synthetic_control/advanced_synthetic_control.html /app/output/docs/4_Advanced/advanced_synthetic_control/
COPY ./project/docs/4_Advanced/advanced_vocalization/advanced_vocalization_draft.html /app/output/docs/4_Advanced/advanced_vocalization/
COPY ./project/docs/4_Advanced/advanced_word_embeddings/advanced_word_embeddings_python_version.html /app/output/docs/4_Advanced/advanced_word_embeddings/
COPY ./project/docs/5_Research/econ490-pystata/01_Setting_Up_PyStata.html /app/output/docs/5_Research/econ490-pystata/
COPY ./project/docs/5_Research/econ490-pystata/02_Working_Dofiles.html /app/output/docs/5_Research/econ490-pystata/
COPY ./project/docs/5_Research/econ490-pystata/09_Stata_Graphs.html /app/output/docs/5_Research/econ490-pystata/
COPY ./project/docs/5_Research/econ490-pystata/11_Linear_Reg.html /app/output/docs/5_Research/econ490-pystata/
COPY ./project/docs/5_Research/econ490-pystata/14_PostReg.html /app/output/docs/5_Research/econ490-pystata/
COPY ./project/docs/5_Research/econ490-r/01_Setting_Up.html /app/output/docs/5_Research/econ490-r/
COPY ./project/docs/5_Research/econ490-r/02_Working_Rscripts.html /app/output/docs/5_Research/econ490-r/
COPY ./project/docs/5_Research/econ490-r/03_R_Essentials.html /app/output/docs/5_Research/econ490-r/
COPY ./project/docs/5_Research/econ490-r/04_Opening_Data_Sets.html /app/output/docs/5_Research/econ490-r/
COPY ./project/docs/5_Research/econ490-r/05_Creating_Variables.html /app/output/docs/5_Research/econ490-r/
COPY ./project/docs/5_Research/econ490-r/07_Combining_Datasets.html /app/output/docs/5_Research/econ490-r/
COPY ./project/docs/5_Research/econ490-r/08_ggplot_graphs.html /app/output/docs/5_Research/econ490-r/
COPY ./project/docs/5_Research/econ490-r/09_Combining_Graphs.html /app/output/docs/5_Research/econ490-r/
COPY ./project/docs/5_Research/econ490-r/10_Linear_Reg.html /app/output/docs/5_Research/econ490-r/
COPY ./project/docs/5_Research/econ490-r/11_Exporting_Output.html /app/output/docs/5_Research/econ490-r/
COPY ./project/docs/5_Research/econ490-r/12_Dummy.html /app/output/docs/5_Research/econ490-r/
COPY ./project/docs/5_Research/econ490-r/13_PostReg.html /app/output/docs/5_Research/econ490-r/
COPY ./project/docs/5_Research/econ490-r/14_Panel_Data.html /app/output/docs/5_Research/econ490-r/
COPY ./project/docs/5_Research/econ490-r/15_Diff_in_Diff.html /app/output/docs/5_Research/econ490-r/
COPY ./project/docs/5_Research/econ490-r/16_IV.html /app/output/docs/5_Research/econ490-r/
COPY ./project/docs/5_Research/econ490-stata/01_Setting_Up.html /app/output/docs/5_Research/econ490-stata/
COPY ./project/docs/5_Research/econ490-stata/02_Working_Dofiles.html /app/output/docs/5_Research/econ490-stata/
COPY ./project/docs/5_Research/econ490-stata/03_Stata_Essentials.html /app/output/docs/5_Research/econ490-stata/
COPY ./project/docs/5_Research/econ490-stata/04_Locals_and_Globals.html /app/output/docs/5_Research/econ490-stata/
COPY ./project/docs/5_Research/econ490-stata/05_Opening_Data_Sets.html /app/output/docs/5_Research/econ490-stata/
COPY ./project/docs/5_Research/econ490-stata/06_Creating_Variables.html /app/output/docs/5_Research/econ490-stata/
COPY ./project/docs/5_Research/econ490-stata/07_Within_Group.html /app/output/docs/5_Research/econ490-stata/
COPY ./project/docs/5_Research/econ490-stata/08_Merge_Append.html /app/output/docs/5_Research/econ490-stata/
COPY ./project/docs/5_Research/econ490-stata/09_Stata_Graphs.html /app/output/docs/5_Research/econ490-stata/
COPY ./project/docs/5_Research/econ490-stata/10_Combining_Graphs.html /app/output/docs/5_Research/econ490-stata/
COPY ./project/docs/5_Research/econ490-stata/11_Linear_Reg.html /app/output/docs/5_Research/econ490-stata/
COPY ./project/docs/5_Research/econ490-stata/12_Exporting_Output.html /app/output/docs/5_Research/econ490-stata/
COPY ./project/docs/5_Research/econ490-stata/13_Dummy.html /app/output/docs/5_Research/econ490-stata/
COPY ./project/docs/5_Research/econ490-stata/14_PostReg.html /app/output/docs/5_Research/econ490-stata/
COPY ./project/docs/5_Research/econ490-stata/15_Panel_Data.html /app/output/docs/5_Research/econ490-stata/
COPY ./project/docs/5_Research/econ490-stata/16_Diff_in_Diff.html /app/output/docs/5_Research/econ490-stata/
COPY ./project/docs/5_Research/econ490-stata/17_IV.html /app/output/docs/5_Research/econ490-stata/
COPY ./project/docs/5_Research/econ490-stata/18_Wf_Guide2.html /app/output/docs/5_Research/econ490-stata/
COPY ./project/docs/6_Projects/projects_example_project_econ325/Projects_Example_Project_ECON325.html /app/output/docs/6_Projects/projects_example_project_econ325/
COPY ./project/docs/6_Projects/projects_example_project_econ326/Projects_Example_Project_ECON326.html /app/output/docs/6_Projects/projects_example_project_econ326/


#Final Stage on lightweight linux
FROM nginx:alpine
COPY --from=builder /app/output /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
