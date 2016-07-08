source /home/julesgm/.bashrc

## Generic node workspace ressource preparation
module load compilers/gcc/4.9 apps/python/2.7.10
source /admin/bin/migrate_softwares.sh $LSCRATCH

## Python virtualenv setup & required modules installation
virtualenv $LSCRATCH/myenv
source $LSCRATCH/myenv/bin/activate
$LSCRATCH/myenv/bin/python -m pip install scipy numpy six matplotlib cython
# The nodes don't have direct access to the internet, so they can't fetch mne directly.
# The other packages are extremely common, and the 'pip' is made to look in a premade local repo.
# To circumvent this, we have saved the github repo with mne-python install source
# at /home/julesgm/COCO/mne_inst/mne-python
$LSCRATCH/myenv/bin/python -m pip install /home/julesgm/COCO/mne_inst/mne-python