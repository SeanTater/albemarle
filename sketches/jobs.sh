#!/bin/bash
email_it_to="stgallag@gmail.com"
while sleep 10; do
  command=$(sqlite3 jobs.db "BEGIN; SELECT command FROM jobs ORDER BY id LIMIT 1; DELETE FROM jobs ORDER BY id LIMIT 1; COMMIT;")
  if test -n "$command"; then
    echo "========= RUNNING $command"
    sh -c "$command"
    echo "========= COMPLETED $command"
    echo
  fi
done
