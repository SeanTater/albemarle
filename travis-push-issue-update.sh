#!/bin/sh
# Update the website

# Include the github ssh key
cat >~/.ssh/config <<EOF
Host github.com
  IdentityFile $PWD/travis-github-automation
EOF

# Use the pages repo
git config user.name "Sean Gallagher"
git config user.email "stgallag@gmail.com"
git fetch gh-pages
git checkout gh-pages

# Get the Github issues
curl -svSLX GET \
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
