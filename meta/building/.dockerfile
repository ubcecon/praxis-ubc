# Taken from COMET will have to be adapted for Praxis
# Build stage 2025-06-04
FROM jlgraves/comet-test:test AS builder

WORKDIR /app

# Copy files from Github
COPY ./meta/building/renv.lock ./project ./

RUN mkdir output

# Quarto render all our documents
RUN quarto render --output-dir /app/output  # Absolute path

# Render specific notebook (to test) with execution and replace the non-executed version
RUN quarto render ./docs/SOCI-415/soci_415_network_analysis.qmd --execute --output-dir /app/output/docs/SOCI-415/

# Final Stage (Added this so it can be ran locally and tested properly)
FROM nginx:alpine
COPY --from=builder /app/output /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
