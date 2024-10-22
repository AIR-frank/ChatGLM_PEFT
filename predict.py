import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from main.evaluation.inferences import inference_with_data_path
from main.predictor.chatglm_lora import Predictor
# /root/code/ChatGLM_PEFT/save_model/qa_dataset/ChatGLM_175746/
pred = Predictor(model_from_pretrained='/root/model/chatglm3-6b/', resume_path='./save_model/qa_dataset/ChatGLM_1111122/')

def batcher(item):
    return pred(**item, max_length=3200, temperature=0, build_message=True)

inference_with_data_path(data_path='qa_dataset', batcher=batcher, save_path='./output.txt', batch_size=32)

# 若你希望能够自行喂入数据, 也可以使用inference_with_data, 注意每一条格式为{"query": "", "history": []}