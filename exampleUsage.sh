#!/usr/bin/env bash

# print help:
./analyze-spots.py -h

# analyze one file:
./analyze-spots.py --draw-plot --save-image exampleData/TT66.BTV.660524@Image@00_57_21_061@1471301834535000000@SPS.USER.HIRADMT1.sdds.gz

# analyze all *.sdds.gz files in a directory
./analyze-spots.py --draw-plot --save-image exampleData/*.sdds.gz
