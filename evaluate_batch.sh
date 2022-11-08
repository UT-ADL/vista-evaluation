CUDA_AVAILABLE_DEVICES=0 nohup python vista_sim/evaluate.py --model _models/ebm-512-s1.onnx --save-video --resize-mode resize --road-width 2.5 --dynamics-model models/dynamics_model_v6_10hz.onnx > runs/$(date +%s)_neuron.txt 2>&1 &
sleep 1s
CUDA_AVAILABLE_DEVICES=1 nohup python vista_sim/evaluate.py --model _models/classifier-512.onnx --wandb --resize-mode resize --road-width 2.5 --dynamics-model models/dynamics_model_v6_10hz.onnx --traces ebm-paper-mae-s2-forward_2022-09-23-10-31-24-resize ebm-paper-mae-s2-backward_2022-09-21-12-34-47-resize > runs/$(date +%s)_neuron.txt 2>&1 &
sleep 1s
CUDA_AVAILABLE_DEVICES=2 nohup python vista_sim/evaluate.py --model _models/mae-s2.onnx --wandb --resize-mode resize --road-width 2.5 --dynamics-model models/dynamics_model_v6_10hz.onnx --traces ebm-paper-mae-s2-forward_2022-09-23-10-31-24-resize ebm-paper-mae-s2-backward_2022-09-21-12-34-47-resize > runs/$(date +%s)_neuron.txt 2>&1 &
sleep 1s
CUDA_AVAILABLE_DEVICES=3 nohup python vista_sim/evaluate.py --model _models/mdn-5-s2.onnx --wandb --resize-mode resize --road-width 2.5 --dynamics-model models/dynamics_model_v6_10hz.onnx --traces ebm-paper-mae-s2-forward_2022-09-23-10-31-24-resize ebm-paper-mae-s2-backward_2022-09-21-12-34-47-resize > runs/$(date +%s)_neuron.txt 2>&1 &
sleep 1s
CUDA_AVAILABLE_DEVICES=1 nohup python vista_sim/evaluate.py --model _models/ebm-spatial-0-s2.onnx --wandb --resize-mode resize --road-width 2.5 --dynamics-model models/dynamics_model_v6_10hz.onnx --traces ebm-paper-mae-s2-forward_2022-09-23-10-31-24-resize ebm-paper-mae-s2-backward_2022-09-21-12-34-47-resize > runs/$(date +%s)_neuron.txt 2>&1 &
sleep 1s
CUDA_AVAILABLE_DEVICES=2 nohup python vista_sim/evaluate.py --model _models/ebm-normal-1-s1.onnx --wandb --resize-mode resize --road-width 2.5 --dynamics-model models/dynamics_model_v6_10hz.onnx --traces ebm-paper-mae-s2-forward_2022-09-23-10-31-24-resize ebm-paper-mae-s2-backward_2022-09-21-12-34-47-resize > runs/$(date +%s)_neuron.txt 2>&1 &


