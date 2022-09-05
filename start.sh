rm nohup.out
#export CUDA_VISIBLE_DEVICES=3
#nohup python -m cProfile -o prof.prof gym_main.py &
#nohup python -u gym_main_custom_nn.py &
#nohup python -u training_policy.py &
nohup python -u vista_plus_pilotnet.py &
#nohup python -u rosbag_to_vista.py &
#nohup python -u searcher.py &
#nohup python -u video_extractor.py &
