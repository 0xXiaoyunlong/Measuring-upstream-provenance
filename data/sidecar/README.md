# data/sidecar/ — underlying sidecar (factory-pattern) materials

These are the minimal frozen inputs the missingness audit is recomputed from. The
factory (fac_*) feature tables are rebuildable from the funding cones by
`../../code/02_evaluate_cross_actor/sidecar/build_sidecar_features.py`, and the
audit itself is run by `../../code/02_evaluate_cross_actor/missingness_audit.py`.

Source mapping. Every file was copied verbatim from the research archive
`研究工程_结构化` (which is not modified).

| package file | source under 研究工程_结构化 |
|---|---|
| sidecar_features_before_recovery.csv | 27_L21_Sidecar增量评估/A0_merge/l21_combined_feature_table.csv |
| sidecar_features_after_recovery.csv | 28_L22_新12组Out边补全与Sidecar重测/A0_merge/l22_sidecar_feature_table_v2.csv |
| recovered_out_edges.csv | 28_L22_.../A0_merge/l22_recovered_out_edges.csv |
| outedge_recovery_status.csv | 28_L22_.../A0_merge/l22_outedge_recovery_status.csv |
| source_log.jsonl | 28_L22_.../A0_merge/l22_source_log.jsonl |
| _real_ledgers.json | 01_冻结档案/04_FACTORY_sidecar/_real_ledgers.json |
| _bean_ledgers.json | 01_冻结档案/04_FACTORY_sidecar/_bean_ledgers.json |
| _audmonk_ledgers.json | 01_冻结档案/04_FACTORY_sidecar/_audmonk_ledgers.json |
| A17_cone_ledger_v2.json | 04_Phi提取/A17_cone_ledger_v2.json |
| l17_cone_edges.csv | 23_L17_Phi预检查与冻结提取/A0_merge/l17_cone_edges.csv |
| l20_step0_feature_table.csv | 26_L20_A18预注册评估执行/A0_merge/l20_step0_feature_table.csv |

Notes: `sidecar_features_before_recovery.csv` is regenerated in place (and verified
byte-equal at the factory-feature cells) by `build_sidecar_features.py` from the
funding cones (_real / _bean / _audmonk ledgers, the A17 Convergence cone, and the
new crews' in-only cones in `l17_cone_edges.csv`), using the sample list in
`l20_step0_feature_table.csv`. `sidecar_features_after_recovery.csv` is the L22
out-edge-recovered version. **Offline-reproducibility scope:** the L22 recovery
re-fetched the missing out-edges from a public Ethereum node (`recovered_out_edges.csv`
lists what was recovered; `outedge_recovery_status.csv` the per-crew status), so the
after-recovery table is shipped as a **frozen artifact** and is not re-crawled here.
Offline, this package reproduces (a) the before-recovery factory features, rebuilt
byte-equal from the cones by `build_sidecar_features.py`, and (b) `frozen
after-recovery table -> the audit -> Figure 4`. The original recovery script is kept
as `../../code/02_evaluate_cross_actor/sidecar/_l22_outedge_source.py` for provenance
(it contains absolute paths and a network call, and is not run by the package). A0
decision docs, round prompts, chat logs and kanban files were NOT copied.
