-- 1. 首先確保切換到正確的數據庫 (根據你導入時的設置，名字可能是 synpuf_omop 或 dsdemo)
USE new_schema; 

-- 2. 提取糖尿病相關代碼 (修復了課件代碼在現代MySQL中的報錯問題)
SELECT 
    co.condition_concept_id, 
    c.concept_name, 
    COUNT(*) as ct 
FROM condition_occurrence co
JOIN concept c ON co.condition_concept_id = c.concept_id
WHERE c.concept_name LIKE '%diabete%'
-- 修正點：必須將 concept_name 也加入分組，否則會報 Error Code: 1055
GROUP BY co.condition_concept_id, c.concept_name 
ORDER BY ct DESC;