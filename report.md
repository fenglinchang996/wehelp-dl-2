# Task Submission

## A

- Total Number of Source Titles: **604894**
- Total Number of Tokenized Titles: **604777**

## B. If A and B are different, what have you done for that?

I filtered out tokens with specific POS tags, punctuation, and whitespace, and removed article titles that resulted in 0 tokens:

- `Caa` (對等連接詞), `Cab` (連接詞：等等), `Cba` (連接詞：的話), `Cbb` (關聯連接詞)
- `P` (介詞), `T` (語助詞), `I` (感嘆詞), `Di` (時態標記)
- `Nh` (代名詞), `DE` (的之得地), `SHI` (是)
- `*CATEGORY` (標點符號), `WHITESPACE` (空白)

## C. Parameters of Doc2Vec Embedding Model

a. Total Number of Training Documents: **604777**
b. Output Vector Size: **100** Min Count: **2** Epochs: **50** Workers: **4**
c. First Self Similarity: **89.10%** Second Self Similarity: **91.20%**

## D. Parameters of Multi-Class Classification Model

a. Arrangement of Linear Layers: **100x100x50x9**
b. Activation Function for Hidden Layers: **ReLU**
c. Activation Function for Output Layers: **Softmax**
d. Loss Function: **Categorical Cross Entropy**
e. Algorithms for Back-Propagation: **SGD (Stochastic Gradient Descent)**
f. Total Number of Training Documents: **483821**
g. Total Number of Testing Documents: **120956**
h. Epochs: **30** Learning Rate: **0.01**
i. Accuracy on Testing Documents: **85.36%**
j. PyTorch Accelerator: **MPS (Apple Silicon)** Batch Size: **64**

## E

1. Change: Used PV-DM with vector size 80 and 300 epochs for Doc2Vec training.
   Result: Even with 300 epochs, the second-self-similarity still decreased (from 91.2% to 76.9%).
2. Change: Used smaller learning rate 0.001 for training classification model.
   Result: The testing accuracy dropped from 85.36% to 81.10% due to the smaller learning rate under the same 30 epochs.
3. Change: Increased training epochs from 30 to 50 for training classification model.
   Result: No significant improvment on the testing accuracy (only slightly increased from 85.36% to 85.71%).
