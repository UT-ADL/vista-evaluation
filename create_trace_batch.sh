# nohup python create_trace.py --resize-mode resize --bag /data/Bolt/bagfiles/ebm-paper-classifier-512-forward_2022-09-23-12-05-48.bag > classifier-512-forward_2022-09-23-12-05-48.out 2>&1 &
# nohup python create_trace.py --resize-mode resize --bag /data/Bolt/bagfiles/ebm-paper-classifier-512-backwards_2022-09-23-11-54-32.bag > classifier-512-backwards_2022-09-23-11-54-32.out 2>&1 &
python create_trace.py --resize-mode resize --bag /data/Bolt/bagfiles/ebm-paper-ebm-512-s1-forward_2022-09-22-11-02-58.bag
python create_trace.py --resize-mode resize --bag /data/Bolt/bagfiles/ebm-paper-ebm-512-s1-backward_2022-09-22-11-13-52.bag
python create_trace.py --resize-mode resize --bag /data/Bolt/bagfiles/ebm-paper-ebm-normal-1-s1-forward_2022-09-22-12-15-22.bag
python create_trace.py --resize-mode resize --bag /data/Bolt/bagfiles/ebm-paper-ebm-normal-1-s1-backward_2022-09-22-12-25-57.bag
python create_trace.py --resize-mode resize --bag /data/Bolt/bagfiles/ebm-paper-ebm-spatial-0-s2-forward_2022-09-23-11-43-09.bag
python create_trace.py --resize-mode resize --bag /data/Bolt/bagfiles/ebm-paper-ebm-spatial-0-s2-backwards_2022-09-23-11-31-39.bag
python create_trace.py --resize-mode resize --bag /data/Bolt/bagfiles/ebm-paper-mae-s2-forward_2022-09-23-10-31-24.bag
python create_trace.py --resize-mode resize --bag /data/Bolt/bagfiles/ebm-paper-mae-s2-backward_2022-09-23-10-19-55.bag
python create_trace.py --resize-mode resize --bag /data/Bolt/bagfiles/ebm-paper-mdn-5-s1-forward_2022-09-21-10-29-09.bag
python create_trace.py --resize-mode resize --bag /data/Bolt/bagfiles/ebm-paper-mdn-5-s1-backward_2022-09-21-10-39-43.bag