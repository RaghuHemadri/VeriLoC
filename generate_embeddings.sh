export HUGGING_FACE_TOKEN='your_token'

#generate module level embeddings
python src/generate_clverilog_embeddings.py --output_dir "output" --embedding_type "module"

#generate line level embeddings
python src/generate_clverilog_embeddings.py --output_dir "output" --embedding_type "line"