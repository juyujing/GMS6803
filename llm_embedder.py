import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import argparse
import os
from tqdm import tqdm

def load_concept_map(data_dir):
    """
    建立 ID -> Concept Name 的映射字典
    """
    print("Loading concept maps...")
    
    # 1. 診斷映射 (feat_conditions.csv)
    # 假設列名是 concept_name (根據你之前的截圖)
    cond_path = os.path.join(data_dir, 'raw', 'feat_conditions.csv')
    if os.path.exists(cond_path):
        cond_df = pd.read_csv(cond_path)
        # 這裡根據你之前的情況，可能是 concept_name
        name_col = 'concept_name' if 'concept_name' in cond_df.columns else 'condition_name'
        if name_col in cond_df.columns:
            cond_map = dict(zip(cond_df.condition_concept_id.astype(str), cond_df[name_col]))
        else:
            print(f"[Warning] 'concept_name' column not found in {cond_path}.")
            cond_map = {}
    else:
        cond_map = {}
    
    # 2. 藥物映射 (feat_drug.csv)
    # 【關鍵修改】直接指定讀取 'drug_name'
    drug_path = os.path.join(data_dir, 'raw', 'feat_drug.csv')
    if os.path.exists(drug_path):
        drug_df = pd.read_csv(drug_path)
        
        # 確認是否存在 drug_name 列
        if 'drug_name' in drug_df.columns:
            print(f"Successfully found 'drug_name' column in {drug_path}.")
            # 建立 ID -> Drug Name 的字典
            drug_map = dict(zip(drug_df.drug_concept_id.astype(str), drug_df['drug_name']))
        else:
            # 容錯處理
            print(f"[Error] 'drug_name' column NOT found. Available columns: {drug_df.columns.tolist()}")
            drug_map = {}
    else:
        print(f"[Error] File not found: {drug_path}")
        drug_map = {}
        
    return cond_map, drug_map

def get_llm_embeddings(text_list, model_name, device):
    print(f"Loading Giant Model: {model_name} on GB-200...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 【GB-200 專屬配置】
    # 1. trust_remote_code=True: 對於 OSS 模型通常需要
    # 2. dtype=torch.bfloat16: Blackwell 架構的最佳精度，節省顯存且不損失精度
    # 3. device_map="auto": 讓 accelerate 自動處理顯存分配，防止 OOM (雖然你顯存夠，但這樣最穩)
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True,
        dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    embeddings = []
    # 20B 模型在 192GB 上可以開大 Batch，這裡設為 16 或 32 比較穩妥，取決於 Sequence Length
    batch_size = 16 
    
    print(f"Encoding {len(text_list)} features with full precision...")
    
    # 只需要推理，不需要梯度
    with torch.no_grad():
        for i in tqdm(range(0, len(text_list), batch_size)):
            batch = text_list[i : i+batch_size]
            
            # Tokenize
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(model.device)
            
            # Forward pass
            outputs = model(**inputs)
            
            # Mean Pooling: 獲取最後一層 Hidden States 的平均值
            # 注意: 如果模型輸出的是 tuple, outputs[0] 通常是 last_hidden_state
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs[0]
                
            # Mask padding tokens (更精確的 Mean Pooling)
            attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * attention_mask, 1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            emb = (sum_embeddings / sum_mask).float().cpu().numpy() # 轉回 float32 存儲
            
            embeddings.append(emb)
            
    return np.vstack(embeddings)

def main(args):
    # 1. 讀取 train.csv 的列名
    print(f"Reading columns from {args.train_csv}...")
    df_head = pd.read_csv(args.train_csv, nrows=1)
    feature_cols = [c for c in df_head.columns if c != 'is_diabetes']
    print(f"Total features to embed: {len(feature_cols)}")
    
    # 2. 準備映射
    cond_map, drug_map = load_concept_map(args.data_dir)
    
    # 3. 生成文本 Prompt
    text_prompts = []
    missing_count = 0
    
    for col in feature_cols:
        if col.startswith('cond_'):
            cid = col.split('_')[1]
            text_prompts.append(cond_map.get(cid, f"Medical condition {cid}"))
            if cid not in cond_map: missing_count += 1
                
        elif col.startswith('drug_'):
            did = col.split('_')[1]
            text_prompts.append(drug_map.get(did, f"Medication {did}"))
            if did not in drug_map: missing_count += 1
                
        elif col == 'age': text_prompts.append("Patient age")
        elif col == 'gender_concept_id': text_prompts.append("Patient gender")
        elif col == 'race_concept_id': text_prompts.append("Patient race")
        else: text_prompts.append("Clinical feature")
            
    if missing_count > 0:
        print(f"[Info] {missing_count} features used placeholders.")
            
    # 4. 獲取 Embeddings
    # 這裡 device 參數傳給 get_llm_embeddings，但因為用了 device_map="auto"，實際由 accelerate 接管
    emb_matrix = get_llm_embeddings(text_prompts, args.llm_model, args.device)
    print(f"Embedding Matrix Shape: {emb_matrix.shape}")
    
    # 5. 保存
    np.save(args.output_path, emb_matrix)
    print(f"Saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset')
    parser.add_argument('--train_csv', type=str, default='dataset/train.csv')
    parser.add_argument('--output_path', type=str, default='dataset/feature_embeddings.npy')
    # 恢復為你的 20B 模型
    parser.add_argument('--llm_model', type=str, default='openai/gpt-oss-20b') 
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)