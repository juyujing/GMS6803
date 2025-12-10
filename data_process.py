import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split

def load_and_pivot(csv_path, index_col, feature_col, value_col, prefix):
    """
    讀取長表格 (Long Format) 並透視為寬表格 (Wide Format)。
    處理邏輯：
    1. 讀取 CSV
    2. 使用 pivot_table 將 (person_id, concept_id) 聚合，將 feature_col 轉為列名
    3. 重置索引，確保 person_id 變回普通列，方便 merge
    """
    if not os.path.exists(csv_path):
        print(f"[Warning] File not found: {csv_path}. Skipping...")
        return None  # 返回 None 表示無數據
        
    df = pd.read_csv(csv_path)
    print(f"Loading {csv_path}: {df.shape}")
    
    # 透視表核心邏輯
    # index: 聚合的主鍵 (person_id)
    # columns: 變成列頭的特徵ID (condition_concept_id / drug_concept_id)
    # values: 填充的值 (condition_count)
    # aggfunc='sum': 如果同一人對同一概念有多條記錄(罕見但可能)，將其相加
    pivot_df = df.pivot_table(
        index=index_col, 
        columns=feature_col, 
        values=value_col, 
        fill_value=0,
        aggfunc='sum' 
    )
    
    # 【關鍵修復】重命名列名，防止特徵衝突，並將 person_id 從索引還原為列
    pivot_df.columns = [f"{prefix}_{col}" for col in pivot_df.columns]
    pivot_df.reset_index(inplace=True) # 這一步讓 person_id 變回 columns 裡的一員
    
    return pivot_df

def process_data(args):
    # 1. 確保輸出目錄存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("=== Start Data Processing ===")

    # 2. 讀取主表 (Cohort)
    cohort_path = os.path.join(args.source_dir, 'cohort_labeled.csv')
    if not os.path.exists(cohort_path):
        raise FileNotFoundError(f"Base cohort file not found at {cohort_path}")
        
    cohort = pd.read_csv(cohort_path)
    print(f"Cohort loaded: {cohort.shape}")

    # 3. 處理特徵並準備合併
    # 3.1 處理診斷 (Conditions)
    cond_path = os.path.join(args.source_dir, 'feat_conditions.csv')
    cond_pivot = load_and_pivot(cond_path, 'person_id', 'condition_concept_id', 'condition_count', 'cond')

    # 3.2 處理藥物 (Drugs)
    # 注意：這裡將文件名從 feat_drug.csv 改為 feat_drugs.csv (根據你的描述可能有的文件名差異，請確保一致)
    # 如果你的文件名是 feat_drug.csv，請保持原樣；如果是 feat_drugs.csv，請修改下面這行
    drug_path = os.path.join(args.source_dir, 'feat_drug.csv') 
    drug_pivot = load_and_pivot(drug_path, 'person_id', 'drug_concept_id', 'drug_count', 'drug')

    # 4. 合併數據 (Left Join)
    df_final = cohort
    
    # 【關鍵修復】只有當 pivot_df 不為 None 時才合併，避免 Key Error
    if cond_pivot is not None:
        print(f"Merging conditions (cols: {cond_pivot.shape[1]})...")
        df_final = df_final.merge(cond_pivot, on='person_id', how='left')
    
    if drug_pivot is not None:
        print(f"Merging drugs (cols: {drug_pivot.shape[1]})...")
        df_final = df_final.merge(drug_pivot, on='person_id', how='left')

    # 填充 NaN 為 0 (Left Join 後產生的空值)
    df_final.fillna(0, inplace=True)
    
    # 5. 移除 ID 列
    if 'person_id' in df_final.columns:
        df_final.drop(columns=['person_id'], inplace=True)

    print(f"Final Data Shape (Pre-split): {df_final.shape}")
    
    # 簡單檢查 Label 分佈
    if 'is_diabetes' in df_final.columns:
        print(f"Class Distribution:\n{df_final['is_diabetes'].value_counts()}")
    else:
        print("[Warning] 'is_diabetes' column not found in cohort file!")
    
    if 'person_id' in df_final.columns:
        df_final.drop(columns=['person_id'], inplace=True)

    # 【新增代碼】強制將 Label 移動到最後一列
    print("Moving 'is_diabetes' to the last column...")
    cols = [c for c in df_final.columns if c != 'is_diabetes'] + ['is_diabetes']
    df_final = df_final[cols]

    # 6. 數據集切分
    # 這裡加入容錯，如果只有一類樣本(比如測試用的假數據)，stratify會報錯
    y = df_final['is_diabetes']
    unique_classes = y.nunique()
    
    if unique_classes > 1:
        # 正常分層切分
        train_df, temp_df = train_test_split(df_final, test_size=0.3, random_state=args.seed, stratify=y)
        tune_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed, stratify=temp_df['is_diabetes'])
    else:
        # 如果只有一類樣本，不使用 stratify
        print("[Warning] Only one class found in targets. Skipping stratification.")
        train_df, temp_df = train_test_split(df_final, test_size=0.3, random_state=args.seed)
        tune_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed)

    # 7. 保存數據
    train_path = os.path.join(args.output_dir, 'train.csv')
    tune_path = os.path.join(args.output_dir, 'tune.csv')
    test_path = os.path.join(args.output_dir, 'test.csv')

    train_df.to_csv(train_path, index=False)
    tune_df.to_csv(tune_path, index=False)
    test_df.to_csv(test_path, index=False)

    # 計算非零元素的比例
    sparsity = (df_final.iloc[:, 5:] != 0).mean().mean()
    print(f"數據稀疏度 (非零佔比): {sparsity:.4%}")

    print(f"=== Processing Done ===")
    print(f"Train: {train_df.shape}, Tune: {tune_df.shape}, Test: {test_df.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默認路徑根據你的報錯信息調整為 ./dataset/raw
    parser.add_argument('--source_dir', type=str, default='./dataset/raw', help='Directory containing raw csv files')
    parser.add_argument('--output_dir', type=str, default='./dataset', help='Directory to save processed datasets')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    process_data(args)