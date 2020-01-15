BOLD := \033[1m
DIM := \033[2m
RESET := \033[0m

help:
	@echo "${BOLD}qac_api:${RESET}	run the question auto-completion API on port 8181"
	@echo "${DIM}		read:	/nfs/students/natalie-prange/language_models/lstm_models/data/aq_gq_cw_combined_shuffled.*"
	@echo "			/nfs/students/natalie-prange/language_models/lstm_models/model/model_100emb_512lu_025drop_512bs_15ep_wd_v8_comb.*"
	@echo "			/nfs/students/natalie-prange/word2vec_models/word2vec_entity_model_wd_200_5_20ep_lm_st_re*"
	@echo "			/nfs/students/natalie-prange/wikidata_mappings/total_file_norm_scores_wd_pure_v8.txt"
	@echo "			/nfs/students/natalie-prange/wikidata_mappings/category_to_sorted_ids_wd_pure_v8.txt"
	@echo "			/nfs/students/natalie-prange/wikidata_mappings/qid_to_aliases_wd_pure.txt"
	@echo "			/nfs/students/natalie-prange/wikidata_mappings/co-occurrence_pair_counts_ents_qid.txt"
	@echo "			/nfs/students/natalie-prange/wikidata_mappings/co-occurrence_pair_counts_ents_whichtype_qid.txt"
	@echo "		time to load: ~ 4 min | required RAM: ~ 14GB${RESET}"


qac_api:
	python3 /home/qac_api.py 80
