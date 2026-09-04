Dataset: Deepsearch-QA
LLM: nvidia/nvidia/nemotron-3-ultra
Judge LLM: gcp/google/gemini-2.5-flash
Agent: Adaptive Researcher
Results Link: 2026-07-31__21-44-45

Number of Queries	Errors	Tier	"Mean F1
(AIQ Harbor Eval)"	"Fully Correct
(AIQ Harbor Eval)"	Avg. Latency (Total) (s)	Avg. Token Usage Input	Avg. Token Usage Output	Avg. No. of LLM calls
90	0	Overall	0.5058	31.11%	401.2	773745.1	16711.5	40.49
0	0	Direct	NA	NA	NA	NA	NA	NA
37	0	Single Shot	0.4129	24.32%	85.6	69812.9	2564.8	4.95
35	0	Standard	0.6203	40%	473.2	818848.2	18179.9	45.29
18	0	Deep	0.4742	27.78%	909.8	2133016.6	42935.6	104.22



Dataset: Deepsearch-QA
LLM: nvidia/nvidia/nemotron-3-ultra
Judge LLM: gcp/google/gemini-2.5-flash
Agent: Shallow Researcher
Results Link: 2026-08-01__03-14-29

Number of Queries	Errors	Intent	"Mean F1
(AIQ Harbor Eval)"	"Fully Correct
(AIQ Harbor Eval)"	Avg. Latency (Total) (s)	Avg. Token Usage Input	Avg. Token Usage Output	Avg. No. of LLM calls
90	2	Overall/Shallow	0.521	28.89%	99.1	64787	2466	4.47



Dataset: Deepsearch-QA
LLM: nvidia/nvidia/nemotron-3-ultra
Judge LLM: gcp/google/gemini-2.5-flash
Agent: Chat Researcher (Intent Router + Shallow + Deep)
Results Link: 2026-07-31__18-57-45

Number of Queries	Errors	Intent	"Mean F1
(AIQ Harbor Eval)"	"Fully Correct
(AIQ Harbor Eval)"	Avg. Latency (Total) (s)	Avg. Token Usage Input	Avg. Token Usage Output	Avg. No. of LLM calls
90	3	Overall	0.6078	42.22%	273.9	305255	8684	17.95
18	3	deep	0.5893	50.00%	958.2	1338911	34530	70.47
72	0	shallow	0.6125	40.28%	131.3	89910	3299	7.01



Dataset: Deepsearch-QA
LLM: nvidia/nvidia/nemotron-3-ultra
Judge LLM: gcp/google/gemini-2.5-flash
Agent: Adaptive Researcher (Standard Only)
Results Link: 2026-08-01__19-18-21

Number of Queries	Errors	Tier	"Mean F1
(AIQ Harbor Eval)"	"Fully Correct
(AIQ Harbor Eval)"	Avg. Latency (Total) (s)	Avg. Token Usage Input	Avg. Token Usage Output	Avg. No. of LLM calls
90	0	Overall	0.5499	31.11%	347.01	825695	20990	47.31
4	0	single_shot	0.65	50.00%	56.61	51328	1924	5.5
86	0	standard	0.5452	30.23%	360.51	861712	21876	49.26



Dataset: Deepsearch-QA
LLM: nvidia/nvidia/nemotron-3-ultra
Judge LLM: gcp/google/gemini-2.5-flash
Agent: Adaptive Researcher (Single shot Only)
Results Link: 2026-08-02__10-40-15

Number of Queries	Errors	Tier	"Mean F1
(AIQ Harbor Eval)"	"Fully Correct
(AIQ Harbor Eval)"	Avg. Latency (Total) (s)	Avg. Token Usage Input	Avg. Token Usage Output	Avg. No. of LLM calls
90	1	Overall	0.4153	28.89%	123.02	245116	9958	14.98
75	1	single_shot	0.429	28.00%	56.74	69752	2988	5.2
14	0	standard	0.3714	35.71%	486.03	1200991	47934	68.29
1	0	meta	0	0.00%	11.8	15187	1066	2



Dataset: Deepsearch-QA
LLM: nvidia/nvidia/nemotron-3-ultra
Judge LLM: gcp/google/gemini-2.5-flash
Agent: Adaptive Researcher (Single shot Only, Increased Tool call limit)
Results Link: 2026-08-03__11-44-56

Number of Queries	Errors	Tier	"Mean F1
(AIQ Harbor Eval)"	"Fully Correct
(AIQ Harbor Eval)"	Avg. Latency (Total) (s)	Avg. Token Usage Input	Avg. Token Usage Output	Avg. No. of LLM calls
76	0	single_shot	0.5828	30.26%	78.14	249463	3996	9
13	0	standard	0.1923	15.38%	493.76	1438874	29739	81.54
1	0	deep	1	100.00%	336.25	1173395	93752	58
90	0	Overall	0.5311	28.89%	141.04	431533	8712	20.02



Dataset: Deepsearch-QA
LLM: nvidia/nvidia/nemotron-3-ultra
Judge LLM: gcp/google/gemini-2.5-flash
Agent: Adaptive Researcher (All tiers enabled, single_shot tool call limit increased)
Results Link:
Iteration: 1

Number of Queries	Errors	Tier	"Mean F1
(AIQ Harbor Eval)"	"Fully Correct
(AIQ Harbor Eval)"	Avg. Latency (Total) (s)	Avg. Token Usage Input	Avg. Token Usage Output	Avg. No. of LLM calls
90	2	Overall	0.4957	25.00%	280.3	680399	14514	35.26
45	0	single_shot	0.5189	26.67%	112.32	275675	4702	9.96
25	1	standard	0.5527	29.17%	357.15	743112	17571	42.75
19	0	deep	0.3685	15.79%	581.05	1559742	33890	85.74
