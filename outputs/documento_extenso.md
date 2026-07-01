<!-- página 8 -->
# Table 3

This table shows evaluation results on multimodal tasks including visual question answering, chart and document understanding. † indicates Chain-of-Thought prompting. All evaluations are 0-shot unless otherwise stated.

| Model | Claude 3 Opus | Claude 3 Sonnet | Claude 3 Haiku | GPT-4Vision | Gemini 1.0 Ultra | Gemini 1.5 Pro | Gemini 1.0 Pro |
|---|---|---|---|---|---|---|---|
| MMMU [3] (val) <br>→ Art & Design | 67.5% | 61.7% | 60.8% | 65.8% | 70.0% | — | — |
| → Business | 67.2% | 58.2% | 52.5% | 59.3% | 56.7% | — | — |
| → Science | 48.9% | 37.1% | 37.1% | 54.7% | 48.0% | — | — |
| → Health & Medicine | 61.1% | 57.1% | 52.3% | 64.7% | 67.3% | — | — |
| → Humanities & Social Science | 70.0% | 68.7% | 66.0% | 72.5% | 78.3% | — | — |
| → Technology & Engineering | 50.6% | 45.0% | 41.5% | 36.7% | 47.1% | — | — |
| Overall | 59.4% | 53.1% | 50.2% | 56.8% (from [3]) | 59.4% | 58.5% | 47.9% |

| DocVQA [53] (test, ANLS score) <br>Document understanding | 89.3% | 89.5% | 88.8% | 88.4% | 90.9% | 86.5% | 88.1% |

| MathVista [54] (testmini) <br>Math | 50.5%† | 47.9%† | 46.4%† | 49.9% (from [54]) | 53% | 52.1% | 45.2% |

| AI2D [52] (test) <br>Science diagrams | 88.1% | 88.7% | 86.7% | 78.2% | 79.5% | 80.3% | 73.9% |

| ChartQA [55] (test, relaxed accuracy) <br>Chart understanding | 80.8%† | 81.1%† | 81.7%† | 78.5%†<br>4-shot | 80.8% | 81.3% | 74.1% |

---  
> **Nota de rodapé:** <br>
† indica Chain-of-Thought prompting.<br>
All evaluations are 0-shot unless otherwise stated.

All GPT scores reported in the GPT-4Vision system card [56], unless otherwise stated.

<!-- página 9 -->
# Human

## What is the average % difference between young adults and elders for G7 nations? Think step by step.

### Younger adults are more likely than their elders to use the internet

| Ages | 40+ | 18-39 | Younger-elder diff |
| --- | --- | --- | --- |
| India | 36% | 73% | +37 |
| Indonesia | 61 | 96 | +35 |
| Poland | 70 | 100 | +30 |
| Hungary | 71 | 98 | +27 |
| Mexico | 70 | 94 | +24 |
| Kenya | 52 | 72 | +20 |
| South Africa | 67 | 86 | +19 |
| Nigeria | 44 | 63 | +19 |
| Japan | 84 | 99 | +15 |
| Argentina | 83 | 98 | +15 |
| Belgium | 87 | 100 | +13 |
| Brazil | 86 | 99 | +13 |
| UK | 88 | 99 | +12 |
| Malaysia | 87 | 100 | +12 |
| Germany | 89 | 100 | +11 |
| Spain | 88 | 99 | +11 |
| Italy | 90 | 100 | +10 |
| France | 86 | 100 | +10 |
| Singapore | 91 | 100 | +9 |
| Canada | 92 | 100 | +8 |
| Greece | 91 | 99 | +8 |
| Netherlands | 94 | 100 | +6 |
| Sweden | 94 | 100 | +6 |
| Israel | 87 | 92 | +5 |
| U.S. | 94 | 98 | +4 |
| South Korea | 98 | 100 | +2 |

---

### Claude 3 Opus

To calculate the average difference in percentage between the 18-39 and 40+ age groups for G7 countries:

1. Identify the G7 countries in the data: Canada, France, Germany, Italy, Japan, UK, USA
2. For each G7 country, subtract the 40+ percentage from the 18-39 percentage:
   - Canada: 100% - 92% = 8%
   - France: 99% - 89% = 10%
   - Germany: 100% - 89% = 11%
   - Italy: 100% - 90% = 10%
   - Japan: 99% - 84% = 15%
   - UK: 100% - 88% = 12%
   - USA: 98% - 94% = 4%

3. Add up those differences:
   8% + 10% + 11% + 10% + 15% + 12% + 4% = 70%

4. Divide the sum by the number of G7 countries (7) to get the average:
   70% / 7 = 10%

The average difference in internet usage percentage between 18-39 year olds and 40+ for G7 countries is 10%.

---

# Figure 1

The figure illustrates an example of Claude 3 Opus's chart understanding combined with multi-step reasoning. We used the chart "Younger adults are more likely than their elders to use the internet" from Pew Research Center [57]. Here the model needed to use its knowledge of G7, identify which countries are G7, retrieve data from the inputted chart and do math using those values.

<!-- página 10 -->
# 5.4 Behavioral Design

Shaping the core behaviors and responses of AI systems to make them safe, ethical, and maximally beneficial to users is a challenging problem in the field that sometimes requires carefully balancing competing objectives. An AI assistant needs to be highly capable and willing to take action to be useful. But it also needs appropriate restraint to avoid misuse. We improved the following areas of behavioral design in the Claude 3 model family: appropriate refusals, honesty and truthfulness, instruction following, and proper formatting for a variety of customer use cases.

## 5.4.1 Refusals

As complexities of model training increase, tradeoffs between helpfulness and harmlessness inevitably arise. Models that are trained to be more helpful and responsive to user requests may also lean towards harmful behaviors (e.g., sharing information that violates our AUP or could be used in dangerous ways). Conversely, models that over index on harmlessness can tend towards not sharing any information with users, even when requests are harmless. Navigating this balancing act is a challenge, and we’ve made good progress on the Claude 3 family, with the models offering fewer refusals to benign prompts.

We developed refusals evaluations to help test the helpfulness aspect of Claude models, measuring where the model unhelpfully refuses to answer a harmless prompt, i.e. where it incorrectly categorizes a prompt as unsafe (violating our AUP) and therefore refuses to answer.

We used the Wildchat dataset [58] for one of our refusal evaluations. This is a collection of diverse user-chatbot interactions that captures a wide range of real-world scenarios, including ambiguous requests, code-switching, topic-switching, and political discussions. One notable aspect of the Wildchat dataset is the presence of toxic user inputs and chatbot responses, which allows for the evaluation of a model’s ability to handle problematic content.

The evaluation process uses both the toxic and non-toxic subsets of the Wildchat dataset. When presented with toxic content, a well-performing model should exhibit a high refusal rate, indicating its ability to identify and reject harmful or inappropriate requests. Conversely, when presented with non-toxic content, the model should have a low refusal rate, demonstrating its capability to engage in harmless conversations and exhibit helpful behavior. As shown in Figure [2], the Claude 3 models demonstrate much more nuanced behavior compared to previous generations of Claude 2, recognizing real harm and refusing to answer harmless prompts much less often.

Additionally, on XSTest evaluation [59], which comprises approximately two hundred non-malicious prompts, the incidence of incorrect refusals by Claude 3 Opus significantly decreased relative to both Claude 2 and other Claude 3 models. Specifically, the refusal rate dropped from 35.1% with Claude 2.1 to just 9%, as illustrated in Figure [3].

To address the issue of over-refusal on benign queries, we further developed a set of internal evaluations based on feedback from customers and users. These evaluations consist of a collection of queries where Claude 2.1 exhibited a tendency to unnecessarily refuse to answer harmless prompts (see Fig. [4]). By analyzing these instances, we established a robust baseline that allowed us to make targeted improvements in the Claude 3 family of models.

We assess our models using two key methods: (1) employing another model to grade responses via few-shot prompts and (2) using string matching to identify refusals. By integrating these methods, we gain a fuller picture of model performance to guide our improvements. To further illustrate the improvements made in the Claude 3 models, we have included additional prompts and their corresponding responses in Appendix [A].