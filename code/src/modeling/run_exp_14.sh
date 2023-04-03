#!/bin/bash

# Change dir
cd /root/co2-flux-hourly-gpp-modeling/

# Run experiment 14_1
#nohup python /root/co2-flux-hourly-gpp-modeling/code/src/modeling/14_1_tft_nogpp_5Y_3D_smallnet_slim.py > /root/co2-flux-hourly-gpp-modeling/data/models/run_logs/14_1_tft_nogpp_5Y_3D_smallnet_slim.log &

# Wait for experiment 14_1 to finish
#wait

# Run experiment 14_2
#nohup python /root/co2-flux-hourly-gpp-modeling/code/src/modeling/14_2_tft_nogpp_5Y_7D_smallnet_slim.py > /root/co2-flux-hourly-gpp-modeling/data/models/run_logs/14_2_tft_nogpp_5Y_7D_smallnet_slim.log &

# Wait for experiment 14_2 to finish
#wait

# Run experiment 14_3
nohup python /root/co2-flux-hourly-gpp-modeling/code/src/modeling/14_3_tft_nogpp_5Y_14D_smallnet_slim.py > /root/co2-flux-hourly-gpp-modeling/data/models/run_logs/14_3_tft_nogpp_5Y_14D_smallnet_slim.log &

# Wait for experiment 14_3 to finish
wait

# Run experiment 14_4
#nohup python /root/co2-flux-hourly-gpp-modeling/code/src/modeling/14_4.py > /root/co2-flux-hourly-gpp-modeling/data/models/run_logs/14_4.log &

# Wait for experiment 14_4 to finish
#wait
