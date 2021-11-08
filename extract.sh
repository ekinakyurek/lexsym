#!/bin/bash
for d in */ ; do
  cd $d
  file=train_encodings.txt
  echo "$file"
  if [ -f "$file" -a ! -f "forward.align.json" ]; then
    echo "$d"
    # awk -F'\t' '{print $1" ||| "$2}' train_encodings.txt  > train_encodings.fast
    # fast_align -i train_encodings.fast -v > forward.align
    # fast_align -i train_encodings.fast -v -r > reverse.align
    # atools -i forward.align -j reverse.align -c intersect > diag.align
    #atools -i forward.align -j reverse.align -c grow-diag > grow-diag.align
    python ~/git/mutex/utils/summarize_aligned_data.py train_encodings.fast ./forward.align
    python ~/git/mutex/utils/summarize_aligned_data.py train_encodings.fast ./reverse.align
    #python ~/git/mutex/utils/summarize_aligned_data.py train_encodings.fast ./grow-diag.align
  fi
  cd ..
done
