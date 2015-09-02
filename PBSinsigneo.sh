#!/bin/bash

## find . -maxdepth 1 -name '*ifun2re*' -delete
## qstat -q insigneo.q
## qrsh -q insigneo.q -P insigneo
## qstat -F | grep insigneo
## qstat | grep insigneo

## submit job:
## qsub -v J=Plots_Closedloop.py,O=ifun -pe ompigige 1 -l rmem=32G -l mem=32G PBSinsigneo.sh
## qsub -v J=Plots_Closedloop.py,O=fig4lruntest -pe ompigige 64 -l rmem=32G -l mem=32G PBSinsigneo.sh

##$ -l h_rt=1:00:00
##$ -l mem=128G
##$ -l rmem=64G

### Queue name
#$ -q insigneo.q
#$ -P insigneo

#$ -l arch=intel*

### Output files
#$ -e log/
#$ -o log/

#$ -j y

module load compilers/intel/12.1.15
module load mpi/intel/openmpi/1.6.4

# Run
echo "======================================================================="

echo $J
echo $O
mpirun python $J -o $O > log/$O.log2 2>&1

echo "======================================================================="
