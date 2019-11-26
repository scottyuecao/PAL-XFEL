# PAL-XFEL
PAL-XFEL data reduction and visualization

The data processing at PAL-XFEL is divided into two steps:
## Data reduction.

All the operations are performed in a terminal.
This reduces the data into tiffs and csvs at each combination of motor positions. Beam stats are recorded in the csv for the laser on and laser off shots respectively.
The area detector background is also produced AFTER the data reduction. The output is an averaged background for a single shot.

## Data visualization

This is performed in an .ipynb file. All the .py files are loaded into the relevant .ipynb files.
