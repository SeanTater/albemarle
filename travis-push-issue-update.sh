#!/bin/sh
# Update the website

# Use the pages repo
git config --global user.name "Sean Gallagher"
git config --global user.email "stgallag@gmail.com"
git clone --branch gh-pages https://SeanTater@github.com/SeanTater/albemarle.git gh-pages
cd gh-pages

# Get the Github issues
curl -svSLX GET \
  -H "Accept: application/vnd.github.v3+json" \
  -H "labels: target" \
  -H "creator: SeanTater" \
  -H "Cache-Control: no-cache" \
  -H "Authorization: token $GITHUB_REPO_KEY" \
  "https://api.github.com/repos/seantater/albemarle/issues" \
  -o _data/issues.json

# This monster turns specific lines from the test output into JSON
# Which has the format [{issue: 1, success:"FAILED"},{issue: 3, success:""}]
egrep "#\d+" ../stacktest.log | perl -pe 's/ +#([0-9]+) (.+ (FAILED) \[\d+\]|.+)\n/\1: "\3"},/' | sed -E 's/(.*),$/{\1}/' >_data/stacktest.json

# Put it on the website
git add _data/issues.json
git add _data/stacktest.json
git commit -m "Updated Github Issue JSON for gh-pages"
git push https://SeanTater:$GITHUB_REPO_KEY@github.com/SeanTater/albemarle.git

# Back to the normal repo
cd ..
