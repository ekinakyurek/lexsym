#!/bin/bash
N=12
(
cd images/
for file in *.svg; do
    ((i=i%N)); ((i++==0)) && wait
    pngf=${file/svg/png}
    if [[ ! -f "$pngf" ]]; then
	svgexport ${file} $pngf 64: &
	
    fi

done
)

