# some-spot-analysis
Image processing for spots in images from SDDS files (uses https://github.com/greyltc/python-sdds)

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
## Example Usage
See [exampleUsage.sh](/exampleUsage.sh)
