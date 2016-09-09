#!/bin/bash
email_it_to="stgallag@gmail.com"
gettime() { date -u +"%Y-%m-%d %H:%M:%S" }
while sleep 10; do
  sqlite3 jobs.db "BEGIN; SELECT id, command FROM jobs ORDER BY id LIMIT 1; DELETE FROM jobs ORDER BY id LIMIT 1; COMMIT;" | read -a command
  if test -n "${command[@]}"; then
    id=${command[0]}
    unset command[0]
    echo "========= RUNNING $command"
    reportf=`mktemp report.XXXXXXXXXXXX`
    start_t="`gettime`"
    sh -c "$command" | tee $reportf
    end_t="`gettime`"
    report="`cat $reportf`"
    rm $reportf
    echo "========= COMPLETED $command"
    sqlite3 jobs.db <<END
      INSERT INTO reports(stdout, start_t, end_t) VALUES ('$id', '$report', '$start_t', '$end_t');
END
    echo
  fi
done
