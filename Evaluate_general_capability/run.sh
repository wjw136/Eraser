python main.py --model peft \
    --model_args pretrained="../models/Llama2-7b-chat-hf/",peft_path="../models/Eraser_Llama2_7b_Lora/"\
    --tasks arc_easy,arc_challenge \
    --batch_size 4 \
    --output_path ./result/Eraser_Llama2_7b.json \
    --device cuda:0

python main.py --model hf-causal \
    --model_args pretrained="../models/Llama2-7b-chat-hf/"\
    --tasks arc_easy,arc_challenge \
    --batch_size 4 \
    --output_path ./result/Original_Llama2_7b.json \
    --device cuda:0