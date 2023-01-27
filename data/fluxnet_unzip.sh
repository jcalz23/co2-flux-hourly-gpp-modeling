#!/bin/bash

mkdir datasets/fluxnet_subset/zip_files

for file in datasets/fluxnet_subset/*; do
    if [[ $file == *"FLUXNET2015_SUBSET"* && $file == *.zip ]]; then
        unzip $file -d ${file%.*}
        mv $file datasets/fluxnet_subset/zip_files/
    fi
done

