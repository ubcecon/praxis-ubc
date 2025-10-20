# Taken from COMET will have to be adapted for Praxis
# Build stage 2025-06-04
FROM jlgraves/comet-test:test AS builder

WORKDIR /app

# Copy files from Github
COPY ./meta/building/renv.lock ./project ./

RUN mkdir output

# Quarto render all documents EXCEPT the problematic one
RUN quarto render --output-dir /app/output

# Copy your pre-rendered HTML file directly (overwriting any static version), can add more here
COPY ./project/_site/docs/SOCI-415/soci_415_network_analysis.html /app/output/docs/SOCI-415/soci_415_network_analysis.html

# Final Stage (Added this so it can be ran locally and tested properly)
FROM nginx:alpine
COPY --from=builder /app/output /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
