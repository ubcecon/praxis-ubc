# Build stage
# Base image is now alexr951/comet-base:safe (Quarto 1.4.557 -> no polyfill.io MathJax shim).
FROM alexr951/comet-base:safe AS builder

WORKDIR /app

# Copy files from Github (places the pre-rendered notebook .html under /app/docs/...)
COPY ./meta/building/renv.lock ./project ./

RUN mkdir output

# Remove the locally-rendered notebooks' .qmd so Quarto skips them (their self-contained
# HTML is placed into the output after render). Collapsed into a SINGLE layer to stay under
# the image layer-depth cap.
RUN rm -f \
    ./docs/4_Advanced/advanced_geospatial/advanced_geospatial.qmd \
    ./docs/4_Advanced/advanced_geospatial/advanced_geospatial_2.qmd \
    ./docs/4_Advanced/advanced_instrumental_variables/advanced_instrumental_variables1.qmd \
    ./docs/4_Advanced/advanced_instrumental_variables/advanced_instrumental_variables2.qmd \
    ./docs/4_Advanced/advanced_linear_differencing/advanced_linear_differencing.qmd \
    ./docs/4_Advanced/advanced_network_analysis/intro_to_python_network_analysis.qmd \
    ./docs/4_Advanced/advanced_network_analysis/network_analysis_notebook.qmd \
    ./docs/4_Advanced/advanced_network_analysis/network_analysis_notebook_II.qmd \
    ./docs/4_Advanced/advanced_synthetic_control/advanced_synthetic_control.qmd \
    ./docs/4_Advanced/advanced_vocalization/advanced_vocalization_draft.qmd \
    ./docs/4_Advanced/advanced_word_embeddings/advanced_word_embeddings_python_version.qmd \
    ./docs/5_Research/econ490-pystata/01_Setting_Up_PyStata.qmd \
    ./docs/5_Research/econ490-pystata/02_Working_Dofiles.qmd \
    ./docs/5_Research/econ490-pystata/09_Stata_Graphs.qmd \
    ./docs/5_Research/econ490-pystata/11_Linear_Reg.qmd \
    ./docs/5_Research/econ490-pystata/14_PostReg.qmd \
    ./docs/5_Research/econ490-pystata/03_Stata_Essentials.qmd \
    ./docs/5_Research/econ490-pystata/04_Locals_and_Globals.qmd \
    ./docs/5_Research/econ490-pystata/05_Opening_Data_Sets.qmd \
    ./docs/5_Research/econ490-pystata/06_Creating_Variables.qmd \
    ./docs/5_Research/econ490-pystata/07_Within_Group.qmd \
    ./docs/5_Research/econ490-pystata/08_Merge_Append.qmd \
    ./docs/5_Research/econ490-pystata/10_Combining_Graphs.qmd \
    ./docs/5_Research/econ490-pystata/12_Exporting_Output.qmd \
    ./docs/5_Research/econ490-pystata/13_Dummy.qmd \
    ./docs/5_Research/econ490-pystata/15_Panel_Data.qmd \
    ./docs/5_Research/econ490-pystata/16_Diff_in_Diff.qmd \
    ./docs/5_Research/econ490-pystata/17_IV.qmd \
    ./docs/5_Research/econ490-pystata/18_Wf_Guide2.qmd \
    ./docs/5_Research/econ490-r/01_Setting_Up.qmd \
    ./docs/5_Research/econ490-r/02_Working_Rscripts.qmd \
    ./docs/5_Research/econ490-r/03_R_Essentials.qmd \
    ./docs/5_Research/econ490-r/04_Opening_Data_Sets.qmd \
    ./docs/5_Research/econ490-r/05_Creating_Variables.qmd \
    ./docs/5_Research/econ490-r/07_Combining_Datasets.qmd \
    ./docs/5_Research/econ490-r/08_ggplot_graphs.qmd \
    ./docs/5_Research/econ490-r/09_Combining_Graphs.qmd \
    ./docs/5_Research/econ490-r/10_Linear_Reg.qmd \
    ./docs/5_Research/econ490-r/11_Exporting_Output.qmd \
    ./docs/5_Research/econ490-r/12_Dummy.qmd \
    ./docs/5_Research/econ490-r/13_PostReg.qmd \
    ./docs/5_Research/econ490-r/14_Panel_Data.qmd \
    ./docs/5_Research/econ490-r/15_Diff_in_Diff.qmd \
    ./docs/5_Research/econ490-r/16_IV.qmd \
    ./docs/5_Research/econ490-stata/01_Setting_Up.qmd \
    ./docs/5_Research/econ490-stata/02_Working_Dofiles.qmd \
    ./docs/5_Research/econ490-stata/03_Stata_Essentials.qmd \
    ./docs/5_Research/econ490-stata/04_Locals_and_Globals.qmd \
    ./docs/5_Research/econ490-stata/05_Opening_Data_Sets.qmd \
    ./docs/5_Research/econ490-stata/06_Creating_Variables.qmd \
    ./docs/5_Research/econ490-stata/07_Within_Group.qmd \
    ./docs/5_Research/econ490-stata/08_Merge_Append.qmd \
    ./docs/5_Research/econ490-stata/09_Stata_Graphs.qmd \
    ./docs/5_Research/econ490-stata/10_Combining_Graphs.qmd \
    ./docs/5_Research/econ490-stata/11_Linear_Reg.qmd \
    ./docs/5_Research/econ490-stata/12_Exporting_Output.qmd \
    ./docs/5_Research/econ490-stata/13_Dummy.qmd \
    ./docs/5_Research/econ490-stata/14_PostReg.qmd \
    ./docs/5_Research/econ490-stata/15_Panel_Data.qmd \
    ./docs/5_Research/econ490-stata/16_Diff_in_Diff.qmd \
    ./docs/5_Research/econ490-stata/17_IV.qmd \
    ./docs/5_Research/econ490-stata/18_Wf_Guide2.qmd

# Quarto render all documents
RUN quarto render --output-dir /app/output

# Place the pre-rendered self-contained HTML (already present under /app/docs from the COPY
# above) into the rendered output, then strip the compromised polyfill.io shim that Quarto
# <1.4 injected into MathJax pages. polyfill.io was taken over by a malicious operator (2024
# supply-chain attack); the shim is unnecessary for MathJax 3 on modern browsers. The base
# image is now Quarto 1.4.557 so freshly rendered pages no longer reference it, but the strip
# stays as a version-independent safety net for the pre-rendered HTML. All in one layer.
RUN mkdir -p /app/output/docs && cd /app/docs && cp --parents \
    4_Advanced/advanced_geospatial/advanced_geospatial.html \
    4_Advanced/advanced_geospatial/advanced_geospatial_2.html \
    4_Advanced/advanced_instrumental_variables/advanced_instrumental_variables1.html \
    4_Advanced/advanced_instrumental_variables/advanced_instrumental_variables2.html \
    4_Advanced/advanced_linear_differencing/advanced_linear_differencing.html \
    4_Advanced/advanced_network_analysis/intro_to_python_network_analysis.html \
    4_Advanced/advanced_network_analysis/network_analysis_notebook.html \
    4_Advanced/advanced_network_analysis/network_analysis_notebook_II.html \
    4_Advanced/advanced_synthetic_control/advanced_synthetic_control.html \
    4_Advanced/advanced_vocalization/advanced_vocalization_draft.html \
    4_Advanced/advanced_word_embeddings/advanced_word_embeddings_python_version.html \
    5_Research/econ490-pystata/01_Setting_Up_PyStata.html \
    5_Research/econ490-pystata/02_Working_Dofiles.html \
    5_Research/econ490-pystata/09_Stata_Graphs.html \
    5_Research/econ490-pystata/11_Linear_Reg.html \
    5_Research/econ490-pystata/14_PostReg.html \
    5_Research/econ490-pystata/03_Stata_Essentials.html \
    5_Research/econ490-pystata/04_Locals_and_Globals.html \
    5_Research/econ490-pystata/05_Opening_Data_Sets.html \
    5_Research/econ490-pystata/06_Creating_Variables.html \
    5_Research/econ490-pystata/07_Within_Group.html \
    5_Research/econ490-pystata/08_Merge_Append.html \
    5_Research/econ490-pystata/10_Combining_Graphs.html \
    5_Research/econ490-pystata/12_Exporting_Output.html \
    5_Research/econ490-pystata/13_Dummy.html \
    5_Research/econ490-pystata/15_Panel_Data.html \
    5_Research/econ490-pystata/16_Diff_in_Diff.html \
    5_Research/econ490-pystata/17_IV.html \
    5_Research/econ490-pystata/18_Wf_Guide2.html \
    5_Research/econ490-r/01_Setting_Up.html \
    5_Research/econ490-r/02_Working_Rscripts.html \
    5_Research/econ490-r/03_R_Essentials.html \
    5_Research/econ490-r/04_Opening_Data_Sets.html \
    5_Research/econ490-r/05_Creating_Variables.html \
    5_Research/econ490-r/07_Combining_Datasets.html \
    5_Research/econ490-r/08_ggplot_graphs.html \
    5_Research/econ490-r/09_Combining_Graphs.html \
    5_Research/econ490-r/10_Linear_Reg.html \
    5_Research/econ490-r/11_Exporting_Output.html \
    5_Research/econ490-r/12_Dummy.html \
    5_Research/econ490-r/13_PostReg.html \
    5_Research/econ490-r/14_Panel_Data.html \
    5_Research/econ490-r/15_Diff_in_Diff.html \
    5_Research/econ490-r/16_IV.html \
    5_Research/econ490-stata/01_Setting_Up.html \
    5_Research/econ490-stata/02_Working_Dofiles.html \
    5_Research/econ490-stata/03_Stata_Essentials.html \
    5_Research/econ490-stata/04_Locals_and_Globals.html \
    5_Research/econ490-stata/05_Opening_Data_Sets.html \
    5_Research/econ490-stata/06_Creating_Variables.html \
    5_Research/econ490-stata/07_Within_Group.html \
    5_Research/econ490-stata/08_Merge_Append.html \
    5_Research/econ490-stata/09_Stata_Graphs.html \
    5_Research/econ490-stata/10_Combining_Graphs.html \
    5_Research/econ490-stata/11_Linear_Reg.html \
    5_Research/econ490-stata/12_Exporting_Output.html \
    5_Research/econ490-stata/13_Dummy.html \
    5_Research/econ490-stata/14_PostReg.html \
    5_Research/econ490-stata/15_Panel_Data.html \
    5_Research/econ490-stata/16_Diff_in_Diff.html \
    5_Research/econ490-stata/17_IV.html \
    5_Research/econ490-stata/18_Wf_Guide2.html \
    /app/output/docs/ \
    && find /app/output -name '*.html' -exec \
    sed -i 's#<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>##g' {} +

# Add the per-notebook launch button (chooses which notebooks get it via launch_notebook.html)
COPY ./meta/building/launch_notebook.html /launch_notebook.html
RUN find /app/output -name '*.html' -exec sh -c \
    'for f; do grep -q "praxis-launch-notebook" "$f" || sed -i "/<body/r /launch_notebook.html" "$f"; done' sh {} +

# Final Stage on lightweight linux
FROM nginx:alpine
COPY --from=builder /app/output /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
