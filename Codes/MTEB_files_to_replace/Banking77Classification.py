"""
/*
* Software Name : mteb
* SPDX-License-Identifier: Apache License 2.0
*
* This software is distributed under the Apache license version 2.0,
* see the "LICENSE.txt" file for more details or https://spdx.org/licenses/Apache-2.0.html
*/
"""
from __future__ import annotations

import time
from collections import defaultdict
import numpy as np
from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class Banking77Classification(AbsTaskClassification):
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
        name="Banking77Classification",
        description="Dataset composed of online banking queries annotated with their corresponding intents.",
        reference="https://arxiv.org/abs/2003.04807",
        dataset={
            "path": "mteb/banking77",
            "revision": "0fd18e25b25c072e09e0d92ab615fda904d66300",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2019-01-01",
            "2019-12-31",
        ),  # Estimated range for the collection of queries
        domains=["Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{casanueva-etal-2020-efficient,
    title = "Efficient Intent Detection with Dual Sentence Encoders",
    author = "Casanueva, I{\~n}igo  and
      Tem{\v{c}}inas, Tadas  and
      Gerz, Daniela  and
      Henderson, Matthew  and
      Vuli{\'c}, Ivan",
    editor = "Wen, Tsung-Hsien  and
      Celikyilmaz, Asli  and
      Yu, Zhou  and
      Papangelis, Alexandros  and
      Eric, Mihail  and
      Kumar, Anuj  and
      Casanueva, I{\~n}igo  and
      Shah, Rushin",
    booktitle = "Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.nlp4convai-1.5",
    doi = "10.18653/v1/2020.nlp4convai-1.5",
    pages = "38--45",
}""",
        prompt="Given a online banking query, find the corresponding intents",
    )
