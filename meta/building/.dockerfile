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

# Quarto render all documents
RUN quarto render --output-dir /app/output

# Place the pre-rendered self-contained HTML (already present under /app/docs from the COPY
# above) into the rendered output, then strip the compromised polyfill.io shim that Quarto
# <1.4 injected into MathJax pages. polyfill.io was taken over by a malicious operator (2024
# supply-chain attack); the shim is unnecessary for MathJax 3 on modern browsers. The base
# image is now Quarto 1.4.557 so freshly rendered pages no longer reference it, but the strip
# stays as a version-independent safety net for the pre-rendered HTML. All in one layer.
RUN mkdir -p /app/output/docs && cd /app/docs && cp --parents \
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
