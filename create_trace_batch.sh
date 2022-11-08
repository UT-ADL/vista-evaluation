# Run this as: `nohup bash create_trace_batch.sh &`

# Option A. run simulatenously (logging to different files):
nohup python create_trace.py --resize-mode resize --bag /data/Bolt/bagfiles/ebm-paper-classifier-512-forward_2022-09-23-12-05-48.bag > classifier-512-forward_2022-09-23-12-05-48.out 2>&1 &
nohup python create_trace.py --resize-mode resize --bag /data/Bolt/bagfiles/ebm-paper-classifier-512-backwards_2022-09-23-11-54-32.bag > classifier-512-backwards_2022-09-23-11-54-32.out 2>&1 &

# Option B. run sequentially (logging into file nohup.out):
python create_trace.py --resize-mode resize --bag /data/Bolt/bagfiles/ebm-paper-classifier-512-forward_2022-09-23-12-05-48.bag
python create_trace.py --resize-mode resize --bag /data/Bolt/bagfiles/ebm-paper-classifier-512-backwards_2022-09-23-11-54-32.bag