# Taken from COMET will have to be adapted for Praxis
# Build stage 2025-06-04
FROM jlgraves/comet-test:test AS builder

WORKDIR /app

# Copy files from Github
COPY ./meta/building/renv.lock ./project ./

RUN mkdir output

# Quarto render all documents (will skip the excluded QMD)
RUN quarto render --output-dir /app/output

# Copy your pre-rendered HTML file to the output
COPY ./project/docs/SOCI-415/soci_415_network_analysis.html /app/output/docs/SOCI-415/

# Final Stage
FROM nginx:alpine
COPY --from=builder /app/output /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
