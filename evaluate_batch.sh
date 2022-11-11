CUDA_AVAILABLE_DEVICES=0 nohup python evaluate.py --model models/ebm-512-s1.onnx --wandb --traces-root /data/Bolt/end-to-end/vista --traces 2021-10-26-10-49-06_e2e_rec_ss20_elva_eval_chunk-resize 2021-10-26-11-08-59_e2e_rec_ss20_elva_back_eval_chunk-resize > runs/$(date +%s)_evaluation.txt 2>&1 &
sleep 1s
CUDA_AVAILABLE_DEVICES=1 nohup python evaluate.py --model models/classifier-512.onnx --wandb --traces-root /data/Bolt/end-to-end/vista --traces ebm-paper-mae-s2-forward_2022-09-23-10-31-24-resize ebm-paper-mae-s2-backward_2022-09-21-12-34-47-resize > runs/$(date +%s)_neuron.txt 2>&1 &
sleep 1s
CUDA_AVAILABLE_DEVICES=2 nohup python evaluate.py --model models/mae-s2.onnx --wandb --traces-root /data/Bolt/end-to-end/vista --traces ebm-paper-mae-s2-forward_2022-09-23-10-31-24-resize ebm-paper-mae-s2-backward_2022-09-21-12-34-47-resize > runs/$(date +%s)_neuron.txt 2>&1 &
sleep 1s
CUDA_AVAILABLE_DEVICES=3 nohup python evaluate.py --model models/mdn-5-s2.onnx --wandb --traces-root /data/Bolt/end-to-end/vista --traces ebm-paper-mae-s2-forward_2022-09-23-10-31-24-resize ebm-paper-mae-s2-backward_2022-09-21-12-34-47-resize > runs/$(date +%s)_neuron.txt 2>&1 &
sleep 1s
CUDA_AVAILABLE_DEVICES=1 nohup python evaluate.py --model models/ebm-spatial-0-s2.onnx --wandb --traces-root /data/Bolt/end-to-end/vista --traces ebm-paper-mae-s2-forward_2022-09-23-10-31-24-resize ebm-paper-mae-s2-backward_2022-09-21-12-34-47-resize > runs/$(date +%s)_neuron.txt 2>&1 &
sleep 1s
CUDA_AVAILABLE_DEVICES=2 nohup python evaluate.py --model models/ebm-normal-1-s1.onnx --wandb --traces-root /data/Bolt/end-to-end/vista --traces ebm-paper-mae-s2-forward_2022-09-23-10-31-24-resize ebm-paper-mae-s2-backward_2022-09-21-12-34-47-resize > runs/$(date +%s)_neuron.txt 2>&1 &


