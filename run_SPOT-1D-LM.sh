#!/bin/bash

python generate_esm_baseline.py --file_list $1 --device $2
python generate_prottrans_baseline.py --file_list $1 --device $3
python run_inference_baseline.py --file_list $1 --device $4
