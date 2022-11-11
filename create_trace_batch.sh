
# Option A. run simulatenously (logging to different files):
# run as: `bash create_trace_batch.sh`
# nohup python create_trace.py --output-root /data/Bolt/end-to-end/vista --bag /data/Bolt/bagfiles/ebm-paper-classifier-512-forward_2022-09-23-12-05-48.bag > classifier-512-forward_2022-09-23-12-05-48.out 2>&1 &
# nohup python create_trace.py --output-root /data/Bolt/end-to-end/vista --bag /data/Bolt/bagfiles/ebm-paper-classifier-512-backwards_2022-09-23-11-54-32.bag > classifier-512-backwards_2022-09-23-11-54-32.out 2>&1 &

# Option B. run sequentially (logging into file nohup.out):
# run as: `nohup bash create_trace_batch.sh &`
# python create_trace.py --output-root /data/Bolt/end-to-end/vista --bag /data/Bolt/bagfiles/ebm-paper-classifier-512-forward_2022-09-23-12-05-48.bag
# python create_trace.py --output-root /data/Bolt/end-to-end/vista --bag /data/Bolt/bagfiles/ebm-paper-classifier-512-backwards_2022-09-23-11-54-32.bag