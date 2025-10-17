"""
/*
* Software Name : mteb
* SPDX-License-Identifier: Apache License 2.0
*
* This software is distributed under the Apache license version 2.0,
* see the "LICENSE.txt" file for more details or https://spdx.org/licenses/Apache-2.0.html
*/
"""
import time
from collections import defaultdict
import numpy as np
from mteb import MTEB
from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from sentence_transformers import SentenceTransformer
from mteb.abstasks.TaskMetadata import TaskMetadata

class SST2BinarySentClassification(AbsTaskClassification):
    def __init__(self, *, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed if seed is not None else int(time.time() * 1000) % (2**32)

    def _undersample_data(self, X, y, samples_per_label, idxs=None):
        X_sampled = []
        y_sampled = []
        
        rng = np.random.default_rng(self.seed)
        if idxs is None:
            idxs = np.arange(len(y))
        rng.shuffle(idxs)

        label_counter = defaultdict(int)
        for i in idxs:
            if label_counter[y[i]] < samples_per_label:
                X_sampled.append(X[i])
                y_sampled.append(y[i])
                label_counter[y[i]] += 1
        return X_sampled, y_sampled, idxs
    
    metadata = TaskMetadata(
        name="SST2BinarySentClassification",
        description="Classification of text movie reviews into positive or negatives sentences.",
        reference="https://huggingface.co/datasets/stanfordnlp/sst2",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["validation"],
        #eval_splits=["validation", "test"],
        eval_langs=["eng"],
        main_score="accuracy",
        dataset={
            "path": "stanfordnlp/sst2",
            "revision": "0.0.0",
        },
        date=("2000-01-01", "2013-10-31"), # best guess
        domains=["Reviews", "Written"],
        task_subtypes=["Emotion classification"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        @inproceedings{socher-etal-2013-recursive,
        title = "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank",
        author = "Socher, Richard  and
        Perelygin, Alex  and
        Wu, Jean  and
        Chuang, Jason  and
        Manning, Christopher D.  and
        Ng, Andrew  and
        Potts, Christopher",
        booktitle = "Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing",
        month = oct,
        year = "2013",
        address = "Seattle, Washington, USA",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/D13-1170",
        pages = "1631--1642",
        }""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"sentence": "text"}
        )