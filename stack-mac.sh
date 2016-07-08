#!/bin/sh
# Use this like `stack` on Mac OSX 
test -d /usr/local/opt/icu4c || echo "the ICU library seems to be missing. Try `brew install icu4c`"
stack --extra-lib-dirs=/usr/local/opt/icu4c/lib --extra-include-dirs=/usr/local/opt/icu4c/include $@
