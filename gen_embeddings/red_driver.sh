VIDEO_PATH=HDfloraine2-320x180.mp4
python3 dim_reduction.py --frame_dir 150_logits/$VIDEO_PATH/ --mode gen_labels --new_size 150 --new_frame_dir labels/$VIDEO_PATH/
python3 dim_reduction.py --frame_dir 4096_embeddings/$VIDEO_PATH/ --new_frame_dir 150_LDA/$VIDEO_PATH/ --mode LDA --new_size 150 --label_dir labels/$VIDEO_PATH/
