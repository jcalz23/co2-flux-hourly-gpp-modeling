#!/bin/bash

cd  ~/co2-flux-hourly-gpp-modeling

python code/src/modeling/FinalExperiments/5-2.RFR-TFT-Lnet-1wk.py  2>&1| tee  data/models/logs-rfr-Lnet1wk.txt
wait

python code/src/modeling/FinalExperiments/6-2.XGBoost-TFT-Lnet-1wk.py  2>&1| tee  data/models/logs-xgb-Lnet1wk.txt
wait

