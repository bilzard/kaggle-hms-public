# 参考文献リスト

## NLL(Noisy Label Learning)

- **Co-Teaching**[1]: 2つのネットワークにより相互にnoisy sampleの情報を教え合う。バッチサンプルのうち、相手のネットワークにおけるラベルとのlossが大きいもの上位1-Rをnoisy labelとして学習サンプルから除外する。
- **Co-Teaching+**[2]: Co-Teachingにおいて、2つのネットワークの独立性を維持するため、双方の予測クラスが異なるサンプルのみを学習サンプルとして利用する。
- **JoCoR**[3]: Co-Teachingと異なり、2つのネットワークを同じloss関数をもとに同時に学習する。また、sample selectionのさい、Co-Teaching+とは逆に、双方のネットワークの予測の確率分布が近いものを優先する。これは「clean labelの予測値の分布は教師ごとのばらつきが小さい」という原則に基づく。

## KD(Knowledge Distillation)

- **CA-MKD**[4]: 複数の教師ネットワークによるKnowledge Distillation(KD)タスクで、教師の予測値のconfidenceをもとに重みづけ平均をとる。また、生徒ネットーワークのembeddingを教師ネットワークのembeddingに近づけるような修飾lossを追加。この手法では「生徒のembeddingを教師のheadに入力した場合のloss」をembeddingの蒸留ラベルの重みとして用いていて、単純に教師の予測値と正解ラベルとのlossを重みとするよりも精度が高かったとのこと。「多少embeddingがずれていても正確な予測ができる教師」を選びやすいはずなので、正確性に加えてロバストさも加味した選別指標と言える。

## GNN(Graph Neural Network)

- **EEG-GNN**[5]
- **Dist/Corr-DCRNN**[6]

## Reference

- [1] [Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels](https://arxiv.org/abs/1804.06872)
- [2] [How does Disagreement Help Generalization against Label Corruption?](https://arxiv.org/abs/1901.04215)
- [3] [Combating noisy labels by agreement: A joint training method with co-regularization](https://arxiv.org/abs/2003.02752)
- [4] [Confidence-Aware Multi-Teacher Knowledge Distillation](https://arxiv.org/abs/2201.00007)
- [5] [EEG-GNN: Graph Neural Networks for Classification of Electroencephalogram (EEG) Signals](https://arxiv.org/abs/2106.09135)
- [6] [Self-Supervised Graph Neural Networks for Improved Electroencephalographic Seizure Analysis](https://arxiv.org/abs/2104.08336)