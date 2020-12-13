
# CS492I_teamUDA

## CV task

* Run main.py without any parameter changes to obtain our current standing best result.
* New Hyperparameters
	* masking: threshold beta for confidence-based masking.
	* sharpening: softmax temperature T for prediction sharpening.
	* useL2loss: default is False, uses L2 loss for UDA loss if True.
	* domain: default is False, uses domain relevance filtering if True.
	* curriculum: default is False, uses curriculum learning if True.
* The best model is stored at kaist0012/fashion_dataset/148/Res18baseMM_best.

## NLP task
* Run run_squad.py without any parameter changes to obtain our best model.
* New Hyperparameters
	* sort_strat: Sorting measure for training context selection. Use "tfidf" and "bm25" for respective values, and any other string for the baseline.
	* query_type: texts to use as the query when sorting training contexts. Use "question" and "answer" to use them individually, or any other string to use both questions and answers.
	* test_cut: If set to true, only looks at the top n contexts sorted by tfidf with the question when choosing test answers.
	* cut_num: value of n for test_cut.
	* use_tokenizer: If set to true, uses the ELECTRA tokenizer when computing tfidf/bm25 measures for training contexts.
* The best model is stored at kaist0012/korquad-open-ldbd3/161/electra_last.
