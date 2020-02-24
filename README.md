# Deep Learning Training with Local Discrete GPU or CPU Integrated GPU instead of CPU
Get to know the workarounds to utilize the Discrete GPU on your PC.
<hr></hr>
<p>
<a href="https://github.com/swastiknath/DL_Training_With_Local_GPU/blob/master/PlaidmlGPU.ipynb"> 1. Obtaining a non-extendable Python Virtual Environment and preparing the GPU for training. </a>
</p>
<hr> </hr

<p> <h3> Working Rule: </h3></p>
<p> We need to obtain the necessary modules like Virtual Environment and PlaidML using the latest versions of Python and Pip. As the documentations of PlaidML mentions these are all experimental features, so it might not work in different environments. Then after creating and activating the venv we will invoke plaidml-setup within the venv to properly configure workload deployment target, which will save its configs into a default dicrectory. Then we just need to install our machine as a backend to our Keras model. </p> 
