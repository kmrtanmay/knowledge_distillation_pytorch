# Knowledge Distillation

Knowledge distillation is model compression method in which a small model is trained to mimic a pre-trained, larger model (or ensemble of models). This training setting is sometimes referred to as "teacher-student", where the large model is the teacher and the small model is the student.

### Dataset: CIFAR-10

## Models
 - Each convolution block is composed of a convolution layer followed by Batch Normalization Layer and ReLU layer.
 - Teacher Model is composed of five convolution blocks followed by two fully connected layers. MaxPool layer is applied at the 1st, 3rd, 5th Convolution blocks.
 - Student Model is composed of three convolution blocks followed by two fully connected layers. MaxPool layer is applied at each Convolution blocks.


 #### Teacher Model(After 10 epochs)
- Train accuracy:82.53%
- Test accuracy  : 67.73%    
  
##  Results(After 10 Epochs)
| Student Models trained          | Train Accuaracy     |  Test Accuracy        | 
| :------------------:            | :----------------:  | :-----------------:   |
| With Knowledge Distillation     | 78.11%              |  67.41%               | 
| Without Knowledge Distillation  | 80.37%              |  67.46%               |

## Reference
H. Li, "Exploring knowledge distillation of Deep neural nets for efficient hardware solutions," [CS230 Report](http://cs230.stanford.edu/files_winter_2018/projects/6940224.pdf), 2018

Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).

Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., & Bengio, Y. (2014). Fitnets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550.

https://github.com/cs230-stanford/cs230-stanford.github.io

https://github.com/bearpaw/pytorch-classification