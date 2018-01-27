#!/bin/bash

clear

for minLineLength in {5..70..5}
do
	for maxGap in {5..55..10}
	do
		for numIntersections in {50..150..25}
		do
			python Img2LnStrng.py \
$maxGap \
$minLineLength \
$numIntersections \
0.9 \
90 \
${minLineLength}_${maxGap}_${numIntersections}_gtf_out.csv \
Vegas.png \
${minLineLength}_${maxGap}_${numIntersections}_Vegas.png
		done
	done
done
