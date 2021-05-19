from __future__ import print_function
import numpy as np
import torch
import os
import torch
from data.label import NodeLabel
from model.logreg import LogReg
import torch.nn as nn
import sklearn
from sklearn.metrics import f1_score
from tqdm import tqdm

def evaluate(embeds, idx_train, idx_val, idx_test, train_lbls, val_lbls, test_lbls, rela=1, cls_test=True, n_runs=50, automl=True):
    hid_units = embeds.shape[1]
    nb_classes = int(max(train_lbls) + 1)
    xent = nn.CrossEntropyLoss()

    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]
    test_ret = {}
    val_ret = {}

    if automl:
        from autosklearn.classification import AutoSklearnClassifier
        train_embs, val_embs, test_embs, train_lbls, val_lbls, test_lbls = [x.cpu().numpy() for x in [train_embs, val_embs, test_embs, train_lbls, val_lbls, test_lbls]]
        n_runs = 1
        print('changing n_runs to 1')

    if cls_test:
        micro_f1s = []
        macro_f1s = []
        macro_f1s_val = [] ##
        micro_f1s_val = []
        device = embeds.device

        for _ in tqdm(range(n_runs), disable=not os.environ.get("TQDM", False) or n_runs == 1):

            if not automl:
                log = LogReg(hid_units, nb_classes, rela=rela)
                opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
                log.to(device)

                val_micro_f1s = []; test_micro_f1s = []
                val_macro_f1s = []; test_macro_f1s = []

                for iter_ in range(50):
                    # train
                    log.train()
                    opt.zero_grad()

                    logits = log(train_embs)
                    loss = xent(logits, train_lbls)

                    loss.backward()
                    opt.step()

                    # val
                    with torch.no_grad():
                        logits = log(val_embs)
                        preds = torch.argmax(logits, dim=1)

                        val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
                        val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

                        val_macro_f1s.append(val_f1_macro)
                        val_micro_f1s.append(val_f1_micro)

                        # test
                        logits = log(test_embs)
                        preds = torch.argmax(logits, dim=1)

                        test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
                        test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

                        test_macro_f1s.append(test_f1_macro)
                        test_micro_f1s.append(test_f1_micro)

                max_iter = val_macro_f1s.index(max(val_macro_f1s))
                val_maf1, test_maf1 = val_macro_f1s[max_iter], test_macro_f1s[max_iter]

                max_iter = val_micro_f1s.index(max(val_micro_f1s))
                val_mif1, test_mif1 = val_micro_f1s[max_iter], test_micro_f1s[max_iter]

            else:
                resampling_strategy = sklearn.model_selection.PredefinedSplit
                _train_embs = np.concatenate((train_embs, val_embs), axis=0)
                _train_lbls = np.concatenate((train_lbls, val_lbls), axis=0)
                resampling_strategy_arguments = {'test_fold': [-1] * len(train_lbls) + [0] * len(val_lbls)}
                model = AutoSklearnClassifier(n_jobs=15, time_left_for_this_task=300, memory_limit=10000, resampling_strategy=resampling_strategy, resampling_strategy_arguments=resampling_strategy_arguments)
                model.fit(_train_embs.copy(), _train_lbls.copy())
                model.refit(train_embs.copy(), train_lbls.copy())
                preds = model.predict(val_embs)
                val_maf1 = f1_score(val_lbls, preds, average='macro')
                val_mif1 = f1_score(val_lbls, preds, average='micro')

                preds = model.predict(test_embs)
                test_maf1 = f1_score(test_lbls, preds, average='macro')
                test_mif1 = f1_score(test_lbls, preds, average='micro')
                del model

            macro_f1s.append(test_maf1)
            macro_f1s_val.append(val_maf1) ###
            micro_f1s.append(test_mif1)
            micro_f1s_val.append(val_mif1)


            
        test_ret.update({
            'micro_f1': np.mean(micro_f1s),
            'macro_f1': np.mean(macro_f1s),
        })
        val_ret.update({
            'micro_f1': np.mean(micro_f1s_val),
            'macro_f1': np.mean(macro_f1s_val),
        })
    return val_ret, test_ret


class NodeEvaluator:

    def __init__(self, graph):
        node_label = NodeLabel(graph)
        self.train_idx = torch.LongTensor(node_label.train_samples)
        self.val_idx = torch.LongTensor(node_label.val_samples)
        self.test_idx = torch.LongTensor(node_label.test_samples)
        self.train_label = torch.LongTensor(node_label.train_labels)[:,0]
        self.val_label = torch.LongTensor(node_label.val_labels)[:,0]
        self.test_label = torch.LongTensor(node_label.test_labels)[:,0]

        self.nb_classes = int(max(self.train_label) + 1)
        print(len(self.train_label), len(self.val_label), len(self.test_label))
        self.results = {}
        self._epoch = 0

    def cuda(self):
        self.train_idx = self.train_idx.cuda()
        self.val_idx = self.val_idx.cuda()
        self.test_idx = self.test_idx.cuda()
        self.train_label = self.train_label.cuda()
        self.val_label = self.val_label.cuda()
        self.test_label = self.test_label.cuda()

    def run(self, emb, rela=1, cls_test=True, n_runs=50):
        val_result, test_result = evaluate(emb, self.train_idx, self.val_idx, self.test_idx, self.train_label, self.val_label, self.test_label, rela=rela, cls_test=cls_test, n_runs=n_runs)
        print('Validation')
        for k, v in val_result.items():
            print('{}: {}'.format(k.upper(), v))
        print('Test')
        for k, v in test_result.items():
            print('{}: {}'.format(k.upper(), v))
        return val_result, test_result