import torch
import pandas as pd
import numpy as np
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.llm_mlp import LLMEnhancedMLP
from dataloader import get_loaders

def load_maps(data_dir):
    """加載 ID -> 名稱 的映射字典，讓 LLM 看懂人話"""
    print("Loading concept maps for translation...")
    
    cond_map = {}
    drug_map = {}
    
    # 1. Condition Map
    cond_path = os.path.join(data_dir, 'raw', 'feat_conditions.csv')
    if os.path.exists(cond_path):
        cond_df = pd.read_csv(cond_path)
        name_col = 'concept_name' if 'concept_name' in cond_df.columns else 'condition_name'
        if name_col in cond_df.columns:
            # 確保 ID 是字符串格式
            cond_map = dict(zip(cond_df.condition_concept_id.astype(str), cond_df[name_col]))
    
    # 2. Drug Map
    drug_path = os.path.join(data_dir, 'raw', 'feat_drug.csv')
    if os.path.exists(drug_path):
        drug_df = pd.read_csv(drug_path)
        # 你的藥物文件列名是 drug_name
        name_col = 'drug_name' if 'drug_name' in drug_df.columns else 'concept_name'
        if name_col in drug_df.columns:
            drug_map = dict(zip(drug_df.drug_concept_id.astype(str), drug_df[name_col]))
    
    print(f"Loaded {len(cond_map)} conditions and {len(drug_map)} drugs mappings.")
    return cond_map, drug_map

def translate_features(feature_list, cond_map, drug_map):
    """將 cond_123 翻譯成醫學術語"""
    translated = []
    for f in feature_list:
        if f.startswith('cond_'):
            cid = f.split('_')[1]
            # 嘗試查找，找不到則保留原樣
            name = cond_map.get(cid, f"Condition_ID_{cid}")
            translated.append(name)
        elif f.startswith('drug_'):
            did = f.split('_')[1]
            name = drug_map.get(did, f"Drug_ID_{did}")
            translated.append(name)
        elif f == 'age':
            translated.append("Patient Age (Elderly)")
        elif f == 'gender_concept_id':
            translated.append("Gender")
        elif f == 'race_concept_id':
            translated.append("Race")
        else:
            translated.append(f)
    return translated

def get_explanation(llm_model, tokenizer, top_features_readable):
    """使用正確的 Chat Template 生成解釋"""
    
    # 定義系統提示詞和用戶提示詞
    messages = [
        {
            "role": "system", 
            "content": "You are an expert medical consultant. Your task is to explain the pathophysiological relationship between risk factors and Diabetes. Be concise, professional, and factual."
        },
        {
            "role": "user", 
            "content": f"""
            Patient Risk Profile:
            The patient has been flagged as high risk for Diabetes based on the following clinical history:
            {', '.join(top_features_readable)}
            
            Question:
            Analyze these specific risk factors. Why do they strongly suggest a diagnosis of Diabetes? 
            Explain the medical reasoning in 1 paragraph.
            """
        }
    ]
    
    # 【關鍵修復】使用 apply_chat_template 處理 Llama-3 格式
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(llm_model.device)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = llm_model.generate(
        input_ids,
        max_new_tokens=4096,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6, # 降低隨機性，防止胡言亂語
        top_p=0.9,
    )
    
    # 只解碼新生成的內容
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response.strip()

def main(args):
    print(f"Loading data on {args.device}...")
    
    # 1. 準備映射
    df_head = pd.read_csv('dataset/train.csv', nrows=1)
    raw_feature_names = [c for c in df_head.columns if c != 'is_diabetes']
    cond_map, drug_map = load_maps('dataset')
    
    _, _, test_loader, input_dim = get_loaders(batch_size=1, num_workers=0)
    
    # 2. 加載 MLP
    print("Loading MLP Model...")
    model = LLMEnhancedMLP(input_dim, args.hidden_dim, 0.0, embedding_path=None).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    
    # 3. 加載 Instruct LLM
    print(f"Loading Explainer: {args.explainer_model} on GB-200...")
    tokenizer = AutoTokenizer.from_pretrained(args.explainer_model)
    
    # GB-200 推薦使用 bfloat16
    llm = AutoModelForCausalLM.from_pretrained(
        args.explainer_model, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print("Generating Reports...")
    count = 0
    output_file = "patient_reports.txt"
    
    with open(output_file, "w") as f:
        with torch.no_grad():
            for inputs, targets in test_loader:
                if count >= args.num_patients: break
                
                inputs = inputs.to(args.device)
                logits = model(inputs)
                prob = torch.sigmoid(logits).item()
                
                # 篩選高風險確診病人
                if prob > 0.85 and targets.item() == 1.0:
                    # 獲取重要特徵
                    weights = model.input_proj.weight.abs().sum(dim=0)
                    contributions = inputs.squeeze() * weights
                    top_indices = torch.topk(contributions, 5).indices.cpu().numpy()
                    top_raw_feats = [raw_feature_names[i] for i in top_indices]
                    
                    # 【關鍵】翻譯成醫學術語
                    readable_feats = translate_features(top_raw_feats, cond_map, drug_map)
                    
                    # 避免重複生成空的翻譯
                    if len(readable_feats) == 0: continue

                    print(f"-> Interpreting Patient {count+1} (Risk: {prob:.2%})...")
                    explanation = get_explanation(llm, tokenizer, readable_feats)
                    
                    report = f"=== Patient Case {count+1} ===\n"
                    report += f"Predicted Risk: {prob:.2%}\n"
                    report += f"Raw Factors: {top_raw_feats}\n" # 保留原始ID方便Debug
                    report += f"Clinical Factors: {readable_feats}\n" # 這是給 LLM 看的
                    report += f"AI Analysis:\n{explanation}\n"
                    report += "-"*50 + "\n"
                    
                    print(report)
                    f.write(report)
                    count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_patients', type=int, default=3)
    # 既然你有GB-200，用這個最強模型
    parser.add_argument('--explainer_model', type=str, default='meta-llama/Llama-3.3-70B-Instruct') 
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)