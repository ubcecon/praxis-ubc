# Taken from COMET will have to be adapted for Praxis
# Build stage 2025-06-04
FROM jlgraves/comet-test:test AS builder

WORKDIR /app

COPY ./meta/building/renv.lock ./project ./

# Remove the original QMD that we have a .html for
RUN rm -f ./docs/SOCI-415/soci_415_network_analysis.qmd

RUN mkdir output

# Quarto render all documents + stub
RUN quarto render --output-dir /app/output

# Copy pre-rendered HTML file
COPY ./project/docs/SOCI-415/soci_415_network_analysis.html /app/output/docs/SOCI-415/

FROM nginx:alpine
COPY --from=builder /app/output /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
