#!/usr/bin/env bash
# Fetch the authors' preprocessed datasets into data/<ds>/handled/ (the paper-comparable setting).
# Downloads the bundle the original LLM-ESR authors shared; to preprocess from raw instead see
# README -> "Preprocessing from raw". Idempotent: does nothing if the data is already present.
#   override:  PY=<python interpreter>
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
FILE_ID=1MpBUjCDLiFIEODTnopSCzDAnS8RzO9aV   # authors' preprocessed bundle (linked in the original README)

if [ -f data/yelp/handled/itm_emb_np.pkl ]; then
  echo "Data already present (data/yelp/handled/) - nothing to do."
  exit 0
fi
command -v gdown >/dev/null 2>&1 || $PY -m pip install --quiet gdown
echo "Downloading the authors' preprocessed bundle (Google Drive id $FILE_ID) ..."
gdown "$FILE_ID" -O /tmp/llmesr_data.zip
echo "Extracting into data/ ..."
unzip -q -o /tmp/llmesr_data.zip -d data/ && rm -f /tmp/llmesr_data.zip
echo "Done. Expected layout: data/<yelp|fashion|beauty>/handled/{inter.txt,itm_emb_np.pkl,usr_emb_np.pkl,pca64_itm_emb_np.pkl,sim_user_100.pkl}"
echo "Next: bash experiments/setup_prereqs.sh   (builds the derived files needed by E6/E9/E13/E14)."
