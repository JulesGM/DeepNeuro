source /home/julesgm/.bashrc
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/lustre2/software-gpu/compilers/pgi/15.10/linux86-64/2015/cuda/7.5/lib64/"

if [[ -z "$LSCRATCH" ]] ; then
	export LSCRATCH="/home/julesgm/COCO/FAKE_SCRATCH/"
fi;

## Generic node workspace ressource preparation
module load "compilers/gcc/4.9" "apps/python/2.7.10" "cuda/7.5.18" "libs/cuDNN/5"
source /admin/bin/migrate_softwares.sh $LSCRATCH

## Python virtualenv setup & required modules installation
virtualenv $LSCRATCH/myenv
source $LSCRATCH/myenv/bin/activate

"$PYTHON" -m pip install -U pip scipy numpy six matplotlib cython

# The nodes don't have direct access to the internet, so they can't fetch mne directly.
# The other packages are extremely common, and the 'pip' is made to look in a premade local repo.
# To circumvent this, we have saved the github repo with mne-python install source
# at /home/julesgm/COCO/mne_inst/mne-python
"$PYTHON" -m pip install /home/julesgm/COCO/mne_inst/mne-python
"$PYTHON" -m pip install /lustre2/software-gpu/apps/python/wheelhouse/tensorflow-0.8.0-cp27-none-linux_x86_64.whl

echo "
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b))
" | "$PYTHON"
