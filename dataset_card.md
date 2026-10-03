---
license: apache-2.0
task_categories:
- text-classification
language:
- en
- es
- fr
- de
- it
- pt
- ja
size_categories:
- 10K<n<100K
source_datasets:
- Shopify/product-catalogue
pretty_name: Shopify Product Category (curated lite)
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.parquet
  - split: validation
    path: data/validation.parquet
  - split: test
    path: data/test.parquet
---

# Shopify Product Category (curated lite)

A curated, balanced subset of [Shopify/product-catalogue](https://huggingface.co/datasets/Shopify/product-catalogue)
for **product categorisation from text**: read a product's title and description, predict its category.

Built as a learning project applying data-curation practice (funnel tracking, label design, leakage checks,
deduplication, rebalancing, text hygiene) to a real public catalogue.

## Task and columns

| Column | Meaning |
|---|---|
| `text` | Model input: product title + description (description truncated to 1000 chars) |
| `summary` | The same product rewritten by an LLM into a standard 4-line English format (see below) |
| `label` | Target: level-2 category, e.g. `Home & Garden > Decor` (141 classes) |
| `full_category` | The original full category path, for reference |

Splits: **20,027 train / 2,498 validation / 2,497 test**, stratified by `label`.

## Curation funnel

| Step | Products |
|---|---|
| raw_train | 38,631 |
| raw_test | 9,658 |
| labelled_train | 38,511 |
| labelled_test | 9,624 |
| merged | 48,135 |
| no_conflicting_titles | 48,106 |
| deduplicated | 48,074 |
| classes_20plus | 47,550 |
| capped | 25,023 |
| train | 20,027 |
| validation | 2,498 |
| test | 2,497 |

## Decisions

- **Label granularity: level 2.** Level 1 is too broad to be useful; at level 3, 4 out of 5 classes had fewer than 20 products.
- **Removed non-product labels** (`Uncategorized`, `Gift Cards`, `Bundles`, `Product Add-Ons`) and 1-level paths.
- **The source's `potential_product_categories` column is excluded.** It contains the correct full path in 100% of products,
  and in 23.5% of products every candidate points to the correct level-2 class. Using it turns *name the category* into *pick from a list*.
- **Dropped 8 titles that appeared with conflicting labels** (29 products; placeholder items such as `example product`).
- **Deduplicated** on normalised title + description after merging the source splits: 32 duplicate products removed.
- **Rebalanced:** classes with fewer than 20 products dropped; classes capped at 500 (random sample).
- **Text hygiene:** removed emails (283 products), URLs (300) and phone numbers (74). A first, looser phone rule
  matched 1,813 products, mostly sizes, years and IDs, and was replaced with a keyword-anchored rule.
- **Languages kept as-is:** mostly English, with Spanish, French, German, Italian, Portuguese and Japanese. (A Day 1 word heuristic estimated ~9% Spanish; it could not see other languages.)
- **Duplicates re-checked after cleaning:** one product became identical to a train product only after text hygiene; it was removed from test. No text appears in more than one split.

## LLM rewrite (`summary` column)

Every product's `text` was rewritten by **gpt-4.1-nano** (OpenAI Batch API, temperature 0) into:

```
Title: ...
Brand: ... (or Unknown)
Description: one sentence (or Unknown)
Details: one sentence on key features (or Unknown)
```

Prompt rules ([prompt v2](https://github.com/raghu619/catalog-curation/blob/main/prompts/rewrite_v2.txt)): **no category line** (the category is the label),
never invent facts, write Unknown when information is missing, copy numbers exactly (decimal commas like `12,5 cm`),
remove IDs, URLs and contact details, translate to English. The 4-line structure is then **enforced in code** (`clean_summary`):
first block only, missing Title taken from the listing's first line.

**Quality, all 25,022 products:**

| Check | Result |
|---|---|
| Bad format | 0 |
| Invented numbers flagged | 358 (1.43%); in 20 hand-labelled cases ~25% were real (rest were unit/year conversions), so roughly 0.35% |
| Description = Unknown | 391 (1.56%); 73 of these had real descriptions (mostly books and plants), a known limitation |
| Median length | 575 -> 260 characters |
| Cost | $0.77 for the full run |

**Prompt history:** v1 invented facts (a `12,5 cm` measurement became "12 sheets") and missed obvious brands.
v2 fixed both. v3 was tested on all products and **rejected**: Description = Unknown rose from 1.56% to 2.50% and flagged invented numbers from 1.43% to 1.73%.

**Train/serve note:** if you train on `summary`, rewrite new products with the same prompt and `clean_summary` before predicting.

## Known limitations

- Capping classes discards about half the usable products and makes test scores reflect a **balanced** catalogue, not the real mix.
- Some source labels are genuinely ambiguous (e.g. an "Airplane Piggy Bank": toy or decor?). This caps achievable accuracy.
- The language estimate is a word-count heuristic, not a language detector.
- About 7% of source descriptions are empty; those products rely on the title alone.

## Source and license

Derived from [Shopify/product-catalogue](https://huggingface.co/datasets/Shopify/product-catalogue), released under Apache-2.0.
This derivative is released under the same license. Images from the source are not included.
