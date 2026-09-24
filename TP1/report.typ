#set page(paper: "a4", margin: (x: 1.35cm, y: 1.15cm), numbering: "1 / 1")
#set text(font: "Libertinus Serif", size: 9pt, lang: "en")
#set par(justify: true, leading: 0.46em)
#set table(inset: 3pt, stroke: 0.35pt)
#show heading.where(level: 1): it => block(above: 0.55em, below: 0.22em)[
  #set text(size: 11pt, weight: "bold")
  #it
]

#align(center)[
  #text(size: 15pt, weight: "bold")[TP1: MLP with PyTorch]
  #linebreak()
  MESSAL Ilyes
]

= Baseline model

*Architecture.* The network follows `784 → 256 → 128 → 10`. Each hidden layer uses ReLU, and the output layer returns one logit for each of the ten classes. The model has 235,146 trainable parameters.

*Loss function.* I used `CrossEntropyLoss` because this is a multiclass classification task with integer class labels. The loss combines log-softmax with negative log-likelihood, so the network does not need a softmax layer.

*Training setup.* The baseline uses Adam with a learning rate of `1e-3`, batches of 64, no regularisation, and 15 epochs. Adam is a useful starting point because it adjusts the update size during training and usually works well with `1e-3` without a long parameter search. Experiment 2 checks this choice against SGD. The images are normalised. The original 60,000 training examples are split into 48,000 training images and 12,000 validation images.

*Results.* The figure _Baseline: Adam lr=1e-3, No Regularization_ ends at 94.12% training accuracy and 88.88% validation accuracy. The final gap is `94.12 - 88.88 = 5.24` percentage points. Training accuracy keeps increasing while validation accuracy levels off near 89%. Validation loss also rises in some epochs while training loss falls. Both patterns indicate overfitting. Test accuracy is 88.40%.

= Experiment 1: model capacity

All four architectures are trained with Adam at `1e-3`, without regularisation, for 15 epochs.

#table(
  columns: (1.35fr, 1fr, 1fr, 1fr, 1fr),
  table.header([*Architecture*], [*Train*], [*Validation*], [*Gap*], [*Test*]),
  [`[4]`], [83.16%], [82.52%], [0.64], [81.90%],
  [`[64]`], [92.01%], [88.66%], [3.35], [88.28%],
  [`[256,128]`], [94.18%], [89.02%], [5.17], [88.47%],
  [`[512,256]`], [94.36%], [88.74%], [5.62], [88.58%],
)

_Exp. 1: Train Accuracy: Architecture_ shows that the larger models learn faster. The `[4]` model compresses 784 pixel values into only four hidden values, so it loses information that helps distinguish similar clothes. Its gap is small because both scores are low, not because it generalises well. It underfits the task.

In _Exp. 1: Val Accuracy: Architecture_, gains become small beyond `[256,128]`. The `[512,256]` model peaks at 89.67% in epoch 11, then finishes at 88.74%. More neurons help fit the training images, but some of the extra detail does not transfer to unseen images. This explains why the final gap grows with capacity: 0.64, 3.35, 5.17, and 5.62 points. I selected `[256,128]`. Its peak validation accuracy is close to that of the larger model, its final validation accuracy is higher, and it uses about half as many parameters.

= Experiment 2: optimiser and learning rate

With SGD, `1e-5` makes updates too small. Validation accuracy reaches only 14.49% after 15 epochs. It reaches 81.74% at `1e-3`. At `0.1`, it finishes at 88.64% and peaks at 89.43% in epoch 12. The curves in _2.A: Val Accuracy: Learning Rate (SGD)_ make the difference clear.

#table(
  columns: (1.4fr, 1fr, 0.65fr, 1.35fr),
  table.header([*Best configuration*], [*Best validation*], [*Epoch*], [*Epochs to 80 / 85 / 88%*]),
  [SGD, `lr=0.1`], [89.43%], [12], [1 / 2 / 7],
  [Adam, `lr=1e-3`], [89.55%], [10], [1 / 1 / 4],
)

Adam converges faster in _2.B: Val Accuracy (15 epochs): Optimiser_. Plain SGD uses the chosen learning rate directly, while Adam adjusts the step for each parameter from recent gradients. This is why Adam can learn quickly at `1e-3`, where SGD finishes at only 81.74%. Adam reaches 89.02% with the same value. It still needs a sensible setting: `0.1` makes the updates unstable and accuracy collapses to 9.45%, while `1e-5` limits validation accuracy to 84.64%.

= Experiment 3: regularisation

#table(
  columns: (1.6fr, 0.72fr, 0.72fr, 0.72fr, 0.72fr, 0.85fr),
  table.header([*Configuration*], [*Train*], [*Val.*], [*Gap*], [*Test*], [*Best epoch*]),
  [No regularisation], [95.43%], [88.92%], [6.51], [88.81%], [10],
  [Dropout 0.3], [90.72%], [88.73%], [1.99], [88.48%], [10],
  [Weight decay `1e-3`], [90.70%], [88.71%], [1.99], [88.27%], [10],
  [Dropout 0.3 + BatchNorm], [91.55%], [89.33%], [2.23], [88.93%], [18],
)

In _Exp. 3: Train-Validation Accuracy Gap: Regularisation_, dropout and weight decay reduce the gap from 6.51 points to about 1.99. Dropout prevents the network from relying on the same neurons for every image. Weight decay keeps the weights from growing too large. Both constraints make memorising the training set harder, which explains the lower training accuracy and smaller gap. BatchNorm keeps hidden activations on a more consistent scale. Combined with dropout, it gives the best final validation accuracy, 89.33%, and the best test accuracy, 88.93%. The model needs until epoch 18 to reach its best validation score because dropout makes each training step noisier.
