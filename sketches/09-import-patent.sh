#!/bin/bash
shopt -s nullglob

while test -n "$1"; do
  list=($1)
  for file in "${list[@]}"; do
    echo "Importing $file"
    sqlite3 patent-sample.db <<EOF
BEGIN;
.mode csv
.import $1 tab
INSERT OR IGNORE INTO patent(number, claims, org, title, abstract)
  SELECT "Patent_No", "Claim_Text", "Organization", "Title", "Abstract"
  FROM tab;
DROP TABLE tab;
COMMIT;
EOF
  done
  shift
done
