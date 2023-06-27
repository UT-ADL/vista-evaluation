#Evaluate a model once per trace
launch_task() {
CUDA_AVAILABLE_DEVICES=$1 python ../evaluate.py --model ../models/$2 --wandb-project rally_2023 --traces-root traces --traces $3 --model-type $4  --road-width=2 
}
launch_task 0 $1 2021-06-07-14-20-07_e2e_rec_ss6-resize $2 &
launch_task 1 $1 2021-06-07-14-36-16_e2e_rec_ss6-resize $2 &
launch_task 2 $1 2021-10-26-10-49-06_e2e_rec_ss20_elva-resize $2 &
launch_task 3 $1 2021-10-26-11-08-59_e2e_rec_ss20_elva_back-resize $2 &
wait
launch_task 0 $1 2022-06-10-13-03-20_e2e_elva_backward-resize $2 &
launch_task 1 $1 2022-06-10-13-23-01_e2e_elva_forward-resize $2 &

