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
#RUN rm -f ./docs/intro_to_cnns/intro_to_cnn.qmd
RUN rm -f ./docs/intro_to_convolutions/intro_to_convolution.qmd
RUN rm -f ./docs/intro_to_deep_learning/intro_to_fundamental_ML.qmd


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
COPY ./project/docs/intro_to_convolution/intro_to_convolutions.html /app/output/docs/intro_to_convolutions/
COPY ./project/docs/intro_to_deep_learning/intro_to_fundamental_ML.html /app/output/docs/intro_to_deep_learning/

#Final Stage on lightweight linux
FROM nginx:alpine
COPY --from=builder /app/output /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
