#!/usr/bin/env bash
trap 'stty sane' EXIT
command="try, $1;, catch e, stack=getReport(e); fprintf(1, '%s\n', stack);, end, exit;"
matlab -nodisplay -nodesktop -nosplash -nojvm -r "$command" | tail -n +11