#!/bin/bash

# Set the log directory
log_dir="/Users/jetcalz07/Desktop/MIDS/W210_Capstone/co2-flux-hourly-gpp-modeling/code/src/EDA/logs"

# Create the log directory if it doesn't exist
if [ ! -d "$log_dir" ]; then
  mkdir "$log_dir"
fi

# Run the first Python script
nohup python gpp_contribution.py &> "$log_dir/gpp_contrib.log" &

# Wait for the first script to complete
wait

# Run the second Python script
nohup python extreme_events.py &> "$log_dir/extreme_events.log" &
