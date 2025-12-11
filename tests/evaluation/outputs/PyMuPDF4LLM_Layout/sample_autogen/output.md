# `AutoGen` **: Enabling Next-Gen LLM Applications via Multi-Agent Conversation** 

**Qingyun Wu** _[†]_ **, Gagan Bansal** _[∗]_ **, Jieyu Zhang** _[±]_ **, Yiran Wu** _[†]_ **, Beibin Li** _[∗]_ **Erkang Zhu** _[∗]_ **, Li Jiang** _[∗]_ **, Xiaoyun Zhang** _[∗]_ **, Shaokun Zhang** _[†]_ **, Jiale Liu** _[∓]_ 

**Ahmed Awadallah** _[∗]_ **, Ryen W. White** _[∗]_ **, Doug Burger** _[∗]_ **, Chi Wang** _[∗]_[1] 

> _∗_ Microsoft Research, _†_ Pennsylvania State University 

> _±_ University of Washington, _∓_ Xidian University 

**==> picture [396 x 125] intentionally omitted <==**

**----- Start of picture text -----**<br>
Conversable agent Plot a chart of META and TESLA  Output:<br>stock price change<br>YTD. $<br>Execute the<br>following code… Month<br>Multi-Agent Conversations Error package  No, please plot %<br>yfinance is not  change!<br>installed<br>Got it! Here is the<br>… … … pip install yfinanceSorry! Please first  revised code …<br>… and then execute the code Output:<br>… … … … Installing… %<br>Joint chat Hierarchical chat<br>Month<br>Agent Customization Flexible Conversation Patterns Example Agent Chat<br>**----- End of picture text -----**<br>


Figure 1: `AutoGen` enables diverse LLM-based applications using multi-agent conversations. (Left) `AutoGen` agents are conversable, customizable, and can be based on LLMs, tools, humans, or even a combination of them. (Top-middle) Agents can converse to solve tasks. (Right) They can form a chat, potentially with humans in the loop. (Bottom-middle) The framework supports flexible conversation patterns. 

## **Abstract** 

`AutoGen`[2] is an open-source framework that allows developers to build LLM applications via multiple _agents_ that can converse with each other to accomplish tasks. `AutoGen` agents are customizable, _conversable_ , and can operate in various modes that employ combinations of LLMs, human inputs, and tools. Using `AutoGen` , developers can also flexibly define agent interaction behaviors. Both natural language and computer code can be used to program flexible conversation patterns for different applications. `AutoGen` serves as a generic framework for building diverse applications of various complexities and LLM capacities. Empirical studies demonstrate the effectiveness of the framework in many example applications, with domains ranging from mathematics, coding, question answering, operations research, online decision-making, entertainment, etc. 

> 1Corresponding author. Email: auto-gen@outlook.com 

> 2 `https://github.com/microsoft/autogen` 

