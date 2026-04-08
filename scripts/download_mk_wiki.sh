#!/usr/bin/env bash
# Download the latest Macedonian Wikipedia dump
set -euo pipefail

DUMP_URL="https://dumps.wikimedia.org/mkwiki/latest/mkwiki-latest-pages-articles.xml.bz2"
OUTPUT_DIR="data/raw/mk"

mkdir -p "$OUTPUT_DIR"

echo "Downloading MK Wikipedia dump (~250MB)..."
wget -c "$DUMP_URL" -O "$OUTPUT_DIR/mkwiki-latest-pages-articles.xml.bz2"
echo "Done: $OUTPUT_DIR/mkwiki-latest-pages-articles.xml.bz2"
