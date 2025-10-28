# Taken from COMET will have to be adapted for Praxis
# Build stage 2025-06-04
FROM jlgraves/comet-test:test AS builder

WORKDIR /app

# Copy files from Github
COPY ./meta/building/renv.lock ./project ./
# Copy Lua filter 
COPY ./meta/scripts/ipynb-title.lua ./
RUN find project -type f -name "*.qmd" -exec dirname {} \; | sort -u | while read dir; do \
      cp ipynb-title.lua "$dir/"; \
    done

RUN mkdir output

# Quarto render all our documents
RUN quarto render --output-dir /app/output  # Absolute path

# Final Stage (Added this so it can be ran locally and tested properly)
FROM nginx:alpine
COPY --from=builder /app/output /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
