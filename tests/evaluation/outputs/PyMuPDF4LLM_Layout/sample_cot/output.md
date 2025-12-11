# **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** 

**Jason Wei** 

**Jason Wei Xuezhi Wang Dale Schuurmans Maarten Bosma Brian Ichter Fei Xia Ed H. Chi Quoc V. Le Denny Zhou** Google Research, Brain Team `{jasonwei,dennyzhou}@google.com` 

## **Abstract** 

We explore how generating a _chain of thought_ —a series of intermediate reasoning steps—significantly improves the ability of large language models to perform complex reasoning. In particular, we show how such reasoning abilities emerge naturally in sufficiently large language models via a simple method called _chain-ofthought prompting_ , where a few chain of thought demonstrations are provided as exemplars in prompting. 

Experiments on three large language models show that chain-of-thought prompting improves performance on a range of arithmetic, commonsense, and symbolic reasoning tasks. The empirical gains can be striking. For instance, prompting a PaLM 540B with just eight chain-of-thought exemplars achieves state-of-the-art accuracy on the GSM8K benchmark of math word problems, surpassing even finetuned GPT-3 with a verifier. 


![](tests/evaluation/outputs/PyMuPDF4LLM_Layout/sample_cot/images/tests/fixtures/sample_cot.pdf-0001-07.png)


**----- Start of picture text -----**<br>
Standard Prompting Chain-of-Thought Prompting<br>Model Input Model Input<br>Q: Roger has 5 tennis balls. He buys 2 more cans of  Q: Roger has 5 tennis balls. He buys 2 more cans of<br>tennis balls. Each can has 3 tennis balls. How many  tennis balls. Each can has 3 tennis balls. How many<br>tennis balls does he have now?
 tennis balls does he have now?<br>A: The answer is 11.
 A: Roger started with 5 balls. 2 cans of 3 tennis balls<br>each is 6 tennis balls. 5 + 6 = 11. The answer is 11.<br>Q: The cafeteria had 23 apples. If they used 20 to<br>make lunch and bought 6 more, how many apples  Q: The cafeteria had 23 apples. If they used 20 to<br>do they have? make lunch and bought 6 more, how many apples<br>do they have?<br>Model Output Model Output<br>A: The answer is 27. A: The cafeteria had 23 apples originally. They used<br>20 to make lunch. So they had 23 - 20 = 3. They<br>bought 6 more apples, so they have 3 + 6 = 9. The<br>answer is 9.<br>**----- End of picture text -----**<br>


Figure 1: Chain-of-thought prompting enables large language models to tackle complex arithmetic, commonsense, and symbolic reasoning tasks. Chain-of-thought reasoning processes are highlighted. 

36th Conference on Neural Information Processing Systems (NeurIPS 2022). 

