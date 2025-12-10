import logging
import os
import random
import numpy as np
import torch

def setup_logger(log_dir, experiment_name):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    
    logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def explain_patient_risk(model, patient_data, feature_names, llm_tokenizer, llm_model):
    """
    model: 訓練好的 MLP
    patient_data: 單個病人的 12846 維向量 (Tensor)
    feature_names: 對應 12846 維的真實文字名稱列表
    """
    # 1. 獲取特徵重要性 (簡單做法：Input * Weight)
    # 對於線性層，Feature Importance ~= Input_Value * Weight_Magnitude
    input_layer_weights = model.input_proj.weight.detach().cpu().numpy() # [Hidden, Input_Dim]
    # 這裡做一個簡化：取 L1 norm 作為權重重要性，或者用 Integrated Gradients
    feature_importance = np.mean(np.abs(input_layer_weights), axis=0) * patient_data.cpu().numpy()
    
    # 2. 找出 Top-K 重要特徵
    top_indices = np.argsort(feature_importance)[-10:] # Top 10
    top_features = [feature_names[i] for i in top_indices if patient_data[i] > 0]
    
    # 3. 構建 Prompt
    prompt = f"""
    You are a medical expert. A machine learning model predicted this patient has a high risk of Diabetes.
    The most contributing clinical features found in this patient's history are:
    {', '.join(top_features)}
    
    Please analyze these features and explain the pathophysiological reasoning why they suggest Diabetes.
    Keep it concise and clinical.
    """
    
    # 4. LLM 生成解釋
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    output = llm_model.generate(**inputs, max_new_tokens=200)
    explanation = llm_tokenizer.decode(output[0], skip_special_tokens=True)
    
    return explanation