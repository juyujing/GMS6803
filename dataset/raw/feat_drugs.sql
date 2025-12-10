USE new_schema;

SELECT 
    de.person_id,
    de.drug_concept_id,
    -- 同樣選出藥名方便你檢查，後續建模主要用 ID
    c.concept_name as drug_name,
    count(*) as drug_count
FROM drug_exposure de
JOIN concept c ON de.drug_concept_id = c.concept_id
GROUP BY de.person_id, de.drug_concept_id, c.concept_name;