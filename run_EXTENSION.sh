python generate_esm_extension.py --file_list $1 --device $2
python generate_proteinbert_extension.py --file_list $1 --device $3
python run_inference_extension.py