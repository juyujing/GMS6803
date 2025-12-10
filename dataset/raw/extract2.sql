USE new_schema; -- 確保使用正確的數據庫

SELECT 
    p.person_id,
    -- 1. 提取人口學特徵 (Demographics)
    (2025 - p.year_of_birth) AS age, -- 計算年齡
    p.gender_concept_id,
    p.race_concept_id,
    
    -- 2. 定義標籤 (Labeling)
    -- 邏輯：如果該病人在 condition_occurrence 表中擁有過 Step 1 列表中的任何一個代碼，則為 1 (Case)，否則為 0 (Control)
    CASE 
        WHEN EXISTS (
            SELECT 1 
            FROM condition_occurrence co
            JOIN concept c ON co.condition_concept_id = c.concept_id
            WHERE co.person_id = p.person_id 
            AND c.concept_name LIKE '%diabete%' -- 這裡再次使用相同的篩選邏輯
        ) THEN 1 
        ELSE 0 
    END AS is_diabetes

FROM person p;