# Adaptive_computing_challenge_2021

To start the model training, refer to the jupyter notebook.
Once the training has been done, transfer the weights of the folder by first creating a new folder called build and then put the trained weights to float_model.

Run python quantization.py --quant_mode calib -b 16 to start the calibration process.

Run python quantization.py --quant_mode test to get the int.xmodel file

Compile the model for kv260 dpu by using vai_c_xir -x build/quant_model/ResNet_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json -o build/quant_model/ -n collision_avoidance

Transfer the required files to the board and run the python app_mt.py code to start Infernce.

Dataset link;https://www.kaggle.com/datasets/kasoarcat/jetbot
