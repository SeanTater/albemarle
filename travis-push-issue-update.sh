#!/bin/sh
# Update the website

# Include the github ssh key
ssh-add travis-github-automation

# Use the pages repo
git checkout gh-pages

# Get the Github issues
curl -v -L -X GET \
  -H "Accept: application/vnd.github.v3+json" \
  -H "labels: target" \
  -H "creator: SeanTater" \
  -H "Cache-Control: no-cache" \
  "https://api.github.com/repos/seantater/albemarle/issues" \
  -o _data/issues.json

# This monster turns specific lines from the test output into JSON
# Which has the format [{issue: 1, success:"FAILED"},{issue: 3, success:""}]
egrep "#\d+" stacktest.log | perl -pe 's/ +#([0-9]+) (.+ (FAILED) \[\d+\]|.+)\n/{issue: \1, success:"\3"},/' | sed -E 's/(.*),$/[\1]/' >_data/stacktest.json

# Put it on the website
git add _data/issues.json
git add _data/stacktest.json
git commit -m "Updated Github Issue JSON for gh-pages"
git push SeanTater@github.com:SeanTater/albemarle.git gh-pages

# Back to the normal repo
git checkout master
