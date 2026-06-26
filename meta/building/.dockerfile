# Taken from COMET will have to be adapted for Praxis
# Build stage 2025-06-04
FROM alexr951/comet-base:safe AS builder

WORKDIR /app

COPY ./meta/building/renv.lock ./project ./

RUN mkdir output

# Quarto render all documents + stub
RUN quarto render --output-dir /app/output

# Strip the compromised polyfill.io shim that Quarto <1.4 injects into MathJax pages.
# polyfill.io was taken over by a malicious operator (2024 supply-chain attack); this is since resolved with a new base-docker image, 
# but still left over as it does not hurt
RUN find /app/output -name '*.html' -exec \
    sed -i 's#<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>##g' {} +

# Add the per-notebook launch button (chooses which notebooks get it via launch_notebook.html)
COPY ./meta/building/launch_notebook.html /launch_notebook.html
RUN find /app/output -name '*.html' -exec sh -c \
    'for f; do grep -q "praxis-launch-notebook" "$f" || sed -i "/<body/r /launch_notebook.html" "$f"; done' sh {} +

#Final Stage on lightweight linux
FROM nginx:alpine
COPY --from=builder /app/output /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
