#PEFT #Adapter

*raw: [arXiv:1902.00751](https://arxiv.org/abs/1902.00751)*  
*reading date: 2026.7.1; Update: 2026.7.1*  
*study timeline pointers:*  
*last: [[LLaMA-Open and Efficient Foundation Language Models]]; next: [[Lora_Low-Rank Adaptation Of Large Language Models]]*



#### Feature Based Transfer
Freeze the entire pre-trained model and use its hidden representations as features for a newly trained task-specific model.

#### Full Parameters Fine Tuning
see [[Improving Language Understanding by Generative Pre-Training]];  

#### Adapter

Add **small** adapter modules to certain parts of the net. When fine-tuning, only change the parameters in these adapters, but **keep the base model untouched**. 
Theoretically, The form of adapter need not to be fixed. And also, adapter modules are not restricted to a fixed location. In principle they can be inserted around almost any differentiable module in a neural network, although their placement significantly affects performance. But practically it often benefits from two features:  
1. **Parameter efficient**. That's the original intention. 
2. **Initially near-identity**. That is, initially, the adapted net should act like the untouched base model. Otherwise, the adapted model may behave randomly and fail to converge while training.  

The main idea of this paper is to propose that by adding small task-specific adapters to a large pre-trained model, it can achieve performance close to full fine-tuning while introducing only a tiny number of additional parameters per task.  
Nevertheless, the paper proposed an adapter architecture on transformer that has been tested very well(vanilla transformer see: [[Attention Is All You Need]]). It added 2 adapter modules separately to a transformer layer, one between the MHA output and its Residual connection layer, the other between the FFN layer and its Residual connection layer. And the architecture of an adapter is a 2-layer bottleneck shaped FFN, first projecting the input down to a low-dimension(far smaller that the input dimension) space, nonlinearizing, and projecting back to its original dimension scale, to limit the number of parameters.  
And in the paper experiments, to make this adapter architecture work, the parameters in LN layers in the transformer layer and the general output layer will also be changed while fine-tuning.  