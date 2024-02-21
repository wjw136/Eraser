cd ./start/

CUDA_VISIBLE_DEVICES=1 python AutoDAN.py \
    --num_steps 20 \
    --model_path "../../models/Llama2-7b-chat-hf/" \
    --lora_path "../../models/Eraser_Llama2_7b_Lora/" \
    --output_path "../output/AutoDAN_Eraser_Llama2_7b.json" \
    --data_path "../datasets/AdvBench.csv"

CUDA_VISIBLE_DEVICES=1 python AutoDAN.py \
    --num_steps 20 \
    --model_path "../../models/Llama2-7b-chat-hf/" \
    --lora_path "" \
    --output_path "../output/AutoDAN_original_Llama2_7b.json" \
    --data_path "../datasets/AdvBench.csv"