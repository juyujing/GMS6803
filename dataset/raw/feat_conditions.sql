USE new_schema;

SELECT 
    co.person_id,
    co.condition_concept_id,
    -- 為了方便查看是什麼病，這裡選出了名字，但在訓練模型時主要用 ID
    c.concept_name,
    count(*) as condition_count
FROM condition_occurrence co
JOIN concept c ON co.condition_concept_id = c.concept_id
WHERE 
    -- 【過濾重點】排除所有糖尿病相關代碼
    c.concept_name NOT LIKE '%diabete%'
GROUP BY co.person_id, co.condition_concept_id, c.concept_name;