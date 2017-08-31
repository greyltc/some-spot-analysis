# some-spot-analysis
image processing for spots in images

## Example Usage
See [exampleUsage.sh](/exampleUsage.sh)

## General Usage
```
$ ./analyze-spots.py -h
usage: analyze-spots.py [-h] [--save-image] [--draw-plot] input [input ...]

Spot analysis on image data taken from SDDS files.

positional arguments:
  input         File(s) to process

optional arguments:
  -h, --help    show this help message and exit
  --save-image  Save data .pgm images to /tmp/pgms/
  --draw-plot   Draw data plot or each file processed
```
