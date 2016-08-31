#!/bin/sh
target="`mktemp $1.XXXXXXXXXXXXXXXXXX`"
cp "$1" "$target"
chmod +x "$target"

./submit.sh <<END
./$target
rm "$target"
END
