cd ./start/

CUDA_VISIBLE_DEVICES=0 python AIM.py \
    --model_path "../../models/Llama2-7b-chat-hf/" \
    --lora_path "../../models/Eraser_Llama2_7b_Lora/" \
    --output_path "../output/AIM_Eraser_Llama2_7b.json" \
    --data_path "../datasets/AdvBench.csv"

CUDA_VISIBLE_DEVICES=0 python AIM.py \
    --model_path "../../models/Llama2-7b-chat-hf/" \
    --lora_path "" \
    --output_path "../output/AIM_original_Llama2_7b.json" \
    --data_path "../datasets/AdvBench.csv"