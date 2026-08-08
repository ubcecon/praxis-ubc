# Build stage
# Base image is now alexr951/comet-base:safe (Quarto 1.4.557 -> no polyfill.io MathJax shim).
FROM alexr951/comet-base:safe AS builder

WORKDIR /app

# Copy project sources (incl. project/_freeze cache) into the build stage
COPY ./meta/building/renv.lock ./project ./

RUN mkdir output

# Quarto render all documents (all notebooks now render from the _freeze cache;
# the pre-rendered-HTML splice workflow is retired). Extra deno heap for the
# largest pages (e.g. econ490-stata 05_Opening_Data_Sets).
ENV QUARTO_DENO_EXTRA_OPTIONS="--v8-flags=--max-old-space-size=8192"
RUN quarto render --output-dir /app/output

# Strip the compromised polyfill.io shim that Quarto <1.4 injected into MathJax pages.
# polyfill.io was taken over by a malicious operator (2024 supply-chain attack); the base
# image is Quarto 1.4.557 so rendered pages no longer reference it, but the strip stays just in case. 
RUN find /app/output -name '*.html' -exec \
    sed -i 's#<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>##g' {} +

# Metadata index for the path visualization page (one JSON fetch). Also gets images from sources directly. 
COPY ./meta/scripts/extract_path_viz_meta.py /extract_pvz_meta.py
RUN python3 /extract_pvz_meta.py /app/output /app/output/pages/pvz_meta.json /app

# Add the per-notebook launch button (chooses which notebooks get it via launch_notebook.html)
COPY ./meta/building/launch_notebook.html /launch_notebook.html
RUN find /app/output -name '*.html' -exec sh -c \
    'for f; do grep -q "praxis-launch-notebook" "$f" || sed -i "/^<body/r /launch_notebook.html" "$f"; done' sh {} +

# Final Stage on lightweight linux
FROM nginx:alpine
COPY --from=builder /app/output /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
