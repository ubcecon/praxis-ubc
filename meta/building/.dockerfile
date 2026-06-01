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

RUN mkdir output

# Quarto render all documents + stub
RUN quarto render --output-dir /app/output

# Strip the compromised polyfill.io shim that Quarto <1.4 injects into MathJax pages.
# polyfill.io was taken over by a malicious operator (2024 supply-chain attack); the
# shim is unnecessary for MathJax 3 on modern browsers. Version-independent safety net.
RUN find /app/output -name '*.html' -exec \
    sed -i 's#<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>##g' {} +

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

#Final Stage on lightweight linux
FROM nginx:alpine
COPY --from=builder /app/output /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
