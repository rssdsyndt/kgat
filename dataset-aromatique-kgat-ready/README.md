# Aromatique KGAT-Ready Dataset

This folder is generated from `dataset-aromatique` for the original KGAT loader assumptions.

Key remapping rule:
- Product entities are remapped to item/entity IDs `0..339`.
- Non-product entities are remapped to IDs `340..997`.
- `train.txt` and `test.txt` contain only product item IDs `< 340`.
- `kg_final.txt` uses the same remapped entity ID space.

Counts:
- entities: 998
- products/items: 340
- relations: 7
- KG triples: 9250
- train interactions: 1490
- test interactions: 406

White Sense:
- old entity ID: 316
- new product ID: 68
- generated local accord triples were added in the source KG before remapping.

Use `entity2id_typed.tsv`, `product2id.tsv`, and `old_to_new_entity_id.tsv` for lookup, retrieval, and explanation.
