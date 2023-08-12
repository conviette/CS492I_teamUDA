
# CS492I_teamUDA

This repository contains the code for the 2020 Fall CS492I course, <Deep Learning for Real-World Problems>. This code improves on the provided baseline code for one CV task and one NLP task.

## CV task
* Task: Image Classification (Shopping Mall Dataset)
* The original baseline code is a basic classification model that uses resnet.
* We make three improvements to this baseline code:
  * Unsupervised Data Augmentation
  * Curriculum Learning
  * Pseudo-labeling

### Code Details
* Run main.py without any parameter changes to obtain our current standing best result.
* New Hyperparameters
	* masking: threshold beta for confidence-based masking.
	* sharpening: softmax temperature T for prediction sharpening.
	* useL2loss: default is False, uses L2 loss for UDA loss if True.
	* domain: default is False, uses domain relevance filtering if True.
	* curriculum: default is False, uses curriculum learning if True.
* The best model is stored at kaist0012/fashion_dataset/148/Res18baseMM_best.

## NLP task
* Task: KorQUAD (Question Answering Dataset)
* The original baseline code is a basic QA model that uses BERT.
* We make three improvements to this baseline code:
  * Using KoELECTRA instead of BERT
  * Using tf-idf and bm25 for better context selection
  * Using bm25 to rank answers for better answer selection

### Code Details
* Run run_squad.py without any parameter changes to obtain our best model.
* New Hyperparameters
	* sort_strat: Sorting measure for training context selection. Use "tfidf" and "bm25" for respective values, and any other string for the baseline.
	* query_type: texts to use as the query when sorting training contexts. Use "question" and "answer" to use them individually, or any other string to use both questions and answers.
	* test_cut: If set to true, only looks at the top n contexts sorted by tfidf with the question when choosing test answers.
	* cut_num: value of n for test_cut.
	* use_tokenizer: If set to true, uses the ELECTRA tokenizer when computing tfidf/bm25 measures for training contexts.
* The best model is stored at kaist0012/korquad-open-ldbd3/161/electra_last.
