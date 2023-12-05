# Landscape of LLMs

![LLM Landscape](./../../images/session_1/part_3_landscape_of_llms/Large%20Language%20Models-Main.png)

# Table of Contents
* [Pretrained LLMs](#-pretrained-llms)
* [Prompt Engineering](#-prompt-engineering)
* [Training LLMs](#-training-llms)
* [Evaluating LLMs](#-evaluating-llms)
* [LLMs Deployment](#-llms-deployment)
* [LLMs Inference Optimisation](#-llms-inference-optimisation)
* [LLMs with Large Context Window](#-llms-with-large-context-window)
* [Challanges with LLMs](#-challanges-with-llms)
* [LLM Applications](#-llm-applications)
* [LLM Application Development](#-llm-application-development)
* [LLM Courses](#-llm-courses)

## 👉 Pretrained LLMs

![Pretrained LLMs](./../../images/session_1/part_3_landscape_of_llms/Large%20Language%20Models-pretrained-llms.png)


### Opensource LLMs

- [Can we stop relying on proprietary LLMs to evaluate open LLMs?](https://www.linkedin.com/posts/danielvanstrien_paper-page-prometheus-inducing-fine-grained-activity-7120763227533139969-ML2K?utm_source=share&utm_medium=member_ios)
    
    `Evaluation` `Open LLM` `Proprietary LLM` `GPT-4` `Feedback Collection dataset` `Prometheus model`
    
    Using proprietary LLMs like GPT-4 to evaluate open LLMs has limitations. The Feedback Collection dataset and the Prometheus model aim to close the gap between open and closed models by providing a way to evaluate open LLMs without relying on proprietary models.
    
- [MosaicML releases MPT-30B, a 30 billion parameter LLM that outperforms GPT-3](https://www.linkedin.com/posts/hagaylupesko_mpt-30b-raising-the-bar-for-open-source-activity-7077673886682603520-O0av?utm_source=share&utm_medium=member_ios)
    
    `LLM` `Open Source` `Machine Learning` `Artificial Intelligence`
    
    MosaicML has released MPT-30B, a 30 billion parameter LLM that outperforms the original GPT-3 175 billion parameter model. It is fully open source for commercial use and comes with two fine-tuned variants: MPT-30B-Instruct and MPT-30B-Chat. MPT-30B-Chat is available to play with on HuggingFace, powered by MosaicML Inference. If you want to start using MPT-30B in production, you can customize and deploy it using MosaicML Training and MosaicML Inference.
    
- [OpenChat Surpasses ChatGPT Performance With Open-Source Model](https://www.linkedin.com/posts/yampeleg_the-first-model-to-beat-100-of-chatgpt-35-activity-7081170084253040640-zM7t?utm_source=share&utm_medium=member_ios)
    
    `LLM` `OpenAI` `ChatGPT` `NLP` `Machine Learning`
    
    OpenChat has developed a new language model, Orca, that outperforms ChatGPT on the Vicuna benchmark. Orca was trained on a smaller dataset than ChatGPT, but achieved better performance by using a more efficient training method. OpenChat has made Orca open-source, so that other researchers can build on its success.
    
- [The Latest Advancements in Large Language Models: Unveiling Llama 2, Code Llama, and More](https://magazine.sebastianraschka.com/p/ahead-of-ai-11-new-foundation-models?utm_campaign=post&utm_medium=web)
    
    `LLM` `Llama 2` `Code Llama` `GPT-4` `OpenAI` `Finetuning` `Transformer-based LLMs` `NeurIPS LLM Efficiency Challenge`
    
    The article discusses the latest advancements in large language models (LLMs), including the release of Meta's Llama 2 and Code Llama models, the leaked GPT-4 model details, OpenAI's new finetuning API, and the NeurIPS LLM Efficiency Challenge. It provides a comprehensive overview of the key features, capabilities, and potential applications of these models, while also highlighting ongoing challenges and debates in the field of LLMs.
    
- [Announcing Mistral 7B: The Most Powerful Language Model For Its Size](https://mistral.ai/news/announcing-mistral-7b/)
    
    `language-models` `machine-learning` `artificial-intelligence`
    
    The Mistral AI team has released Mistral 7B, a 7.3B parameter language model that outperforms Llama 2 13B on all metrics. It is easy to fine-tune on any task and is released under the Apache 2.0 license.
    
- [Hugging Face Unveils Zephyr-7b: A State-of-the-Art 7B Chatbot](https://www.linkedin.com/posts/ed-beeching-3553b468_we-will-soon-release-the-hugging-face-llm-activity-7117524096623484929-i6Pa?utm_source=share&utm_medium=member_ios)
    
    `LLM` `Chatbot` `Natural Language Processing` `Artificial Intelligence`
    
    Hugging Face has released Zephyr-7b, a 7B chatbot that outperforms other models in its class on the MT Bench and Open LLM Leaderboard. The model was trained using a combination of instruction fine-tuning and Direct Preference Optimization on publicly available datasets. It is available to try out on the Hugging Face website.
    
- [LaMini-LM: Can Small Language Models Compete with Large Ones?](https://levelup.gitconnected.com/no-gpu-ok-this-mini-but-decent-language-model-can-run-on-your-obsolete-computer-540abf0e2b5b)
    
    `language models` `parameter scale` `computational requirements` `LaMini-LM` `distilled instructions`
    
    LaMini-LM is a small language model with a huge amount of distilled instructions. It is designed to achieve impressive results with a smaller model locally. In this article, we will delve into the details of LaMini-LM and see how tiny computational requirements the model asks for.
    
- [Open Source LLaMA 13B Released with Full Commercial Usage Rights](https://www.linkedin.com/posts/sanyambhutani_fully-open-source-llama-13b-is-here-activity-7076550805570355201-RXhq?utm_source=share&utm_medium=member_ios)
    
    `Open Source LLaMA` `RedPajama Dataset` `SlimPajama Dataset` `Code Generation` `Commercial Usage` `Energy Efficiency`
    
    OpenLM research has released a fully open source version of the LLaMA 13B model, trained on the RedPajama dataset. The model weights are available in both Jax and PyTorch. The model is not ideal for code generation due to its treatment of empty spaces, but it remains one of the best open source models for building on top of. The authors are considering training future releases on the SlimPajama dataset, which is a cleaned version of the RedPajama dataset with 49% smaller size.
    
- [Meet Notus-7B: Data Curation and Open Science go a long way in shaping AI's future](https://argilla.io/blog/notus7b/)
    
    `Open Source LLM` `RLHF` `DPO`
    
    LLama 1 & 2 opened the floodgates of open source LLMs. MistralAI released the most powerful 7B base LLM remotely inspired by the success of LLama 2. HuggingFace H4 released Zephyr trained on on a mix of publicly available, synthetic datasets using DPO. TsinghuaNLP released the UltraChat dataset, a large-scale, multi-round dialogue dataset. OpenBMB released the UltraFeedback dataset, a large-scale, fine-grained, diverse preference dataset for RLHF and DPO. Huggingface H4 team fine-tuned Zephyr using UltraChat (supervised fine tuning) and UltraFeedback (DPO for alignment). ArgillaIO fixed some data issues and improved on Zephyr to release Notus-7B.

## 👉 Prompt Engineering

![Prompt Engineering](./../../images/session_1/part_3_landscape_of_llms/Large%20Language%20Models-prompt-engineering.png)

- [Prompt Engineering Guide](https://www.promptingguide.ai/)
    
    `Prompt Engineering` 
    
    Prompt engineering is a relatively new discipline for developing and optimizing prompts to efficiently use language models (LMs) for a wide variety of applications and research topics. Prompt engineering skills help to better understand the capabilities and limitations of large language models (LLMs).

## 👉 Training LLMs

![Training LLMs](./../../images/session_1/part_3_landscape_of_llms/Large%20Language%20Models-training.png)

- [Efficient Deep Learning Optimization Libraries for Large Language Model Training](https://www.linkedin.com/posts/ashishpatel2604_datascience-machinelearning-artificialintelligence-activity-7082215572150587392-SPN-?utm_source=share&utm_medium=member_ios)
    
    `DeepSpeed` `Megatron-DeepSpeed` `FairScale` `Megatron-LM` `Colossal-AI` `BMTrain` `Mesh TensorFlow` `max text` `Alpa` `GPT-NeoX`
    
    This article provides an overview of various deep learning optimization libraries that can simplify and optimize the training process for large language models. These libraries offer features such as distributed training, model parallelism, and efficient training algorithms, enabling researchers and practitioners to achieve better results with less effort.
    
- [LLM Training Techniques](https://www.linkedin.com/posts/bhavsarpratik_fine-tuning-llms-best-practices-and-when-activity-7077291574581166081-_FBV?utm_source=share&utm_medium=member_ios)
    
    `Training vs Prompting Engineering` `Task Diversity for OOD Robustness` `Self-Instruction for Dataset Generation` `Self-Consistency for Higher Performance` `Evaluation`
    
    This MLOps Community podcast with Mark Huang discusses various LLM training techniques, including training vs prompting engineering, task diversity for OOD robustness, self-instruction for dataset generation, self-consistency for higher performance, and evaluation.
    
- [Deploying RLHF with 0 Annotations: A Case Study](https://www.linkedin.com/posts/prithivirajdamodaran_copy-my-0-annotation-rlhf-strategy-activity-7081134400792322048-BiIQ?utm_source=share&utm_medium=member_ios)
    
    `real-world case-study` `reducing manual effort` `RLHF` `translation quality` `reward model` `user-designated pair` `regression model` `Allen AI's library RL4LMs` `T5/Flan-T5` `HF Trainer` `Sentence Transformers Cross-Encoders`
    
    This article presents a real-world case study of deploying RLHF with 0 annotations. It describes the challenges faced by a large translation company in SE Asia, and how RLHF was used to reduce manual effort in producing domain-specific vocabulary and robotic translations. The article also discusses the tools and libraries used, and provides a key takeaway for readers.
    
- [X-LLM: A Framework for Training Multimodal Language Models](https://www.linkedin.com/posts/ashishpatel2604_llms-data-analytics-activity-7079043798378422272-SpTX?utm_source=share&utm_medium=member_ios)
    
    `Multimodal Language Models` `X-LLM` `Image Captioning` `Text-to-Speech` `Multimodal Question Answering`
    
    The paper proposes a new framework, X-LLM, for training multimodal language models. X-LLM consists of three main components: single-modal encoders, X2L interfaces, and a large language model (LLM). The authors evaluate X-LLM on a variety of tasks and show that it achieves state-of-the-art results.
    
- [TRL: A Full-Stack Transformer Language Model with Reinforcement Learning](https://github.com/huggingface/trl)
    
    `Reinforcement Learning` `Transformer Language Models` `Supervised Fine-tuning` `Reward Modeling` `Proximal Policy Optimization`
    
    TRL is a full-stack library that provides tools for training transformer language models and stable diffusion models with Reinforcement Learning. It is built on top of the transformers library by 🤗 Hugging Face and supports most decoder and encoder-decoder architectures.
    
- [Is Reinforcement Learning Really Necessary for Large Language Models?](https://www.linkedin.com/posts/yoelzeldes_to-get-llms-as-good-as-openais-gpt-4-is-activity-7078958558519656451-N6Wo?utm_source=share&utm_medium=member_ios)
    
    `language models` `reinforcement learning` `direct preference optimization`
    
    The paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" introduces a novel algorithm that gets rid of the two stages of RL, namely - fitting a reward model, and training a policy to optimize the reward via sampling. This new algorithm, called Direct Preference Optimization (DPO), trains the LLM using a new loss function which encourages it to increase the likelihood of the better completion and decrease the likelihood of the worse completion. DPO has been shown to achieve comparable performance to RL-based methods, but is much simpler to implement and scale.
    

### Supervised Finetuning

- [Fine-tuning Llama-2 on your own data](https://www.linkedin.com/posts/alphasignal_llama-2-can-now-be-fine-tuned-on-your-activity-7116422223191576576-ZPb5?utm_source=share&utm_medium=member_ios)
    
    `LLM` `Fine-tuning` `Natural Language Processing`
    
    The new script allows for fine-tuning Llama-2 on your own data in just a few lines of code. It handles single/multi-gpu and can even be used to train the 70B model on a single A100 GPU by leveraging 4bit.
    
- [Fine-tuning LLMs for specific tasks](https://www.linkedin.com/posts/llamaindex_shunyu-yao-on-x-activity-7117924606144901120-Si5v?utm_source=share&utm_medium=member_ios)
    
    `LLM` `fine-tuning` `performance`
    
    The author of the ReAct paper explores the effects of fine-tuning LLMs on specific tasks. They found that fine-tuning significantly improves performance when using the LLM as an agent. The key is to fine-tune each module to tailor it to specific tasks.
    
- [A discussion on various LLM fine-tuning techniques](https://www.linkedin.com/posts/prithivirajdamodaran_%3F%3F%3F%3F-%3F%3F%3F%3F%3F%3F%3F%3F%3F-%3F%3F%3F%3F%3F%3F-activity-7111673119831937024-3Ffu?utm_source=share&utm_medium=member_ios)
    
    `lora` `adapter` `prompt tuning` `rl based policy finetuning`
    
    The post discusses various LLM fine-tuning techniques. It covers LORA, adapters, prompt tuning and RL based policy finetuning. The discussion revolves around the advantages and disadvantages of each technique and the scenarios where they are most suitable.
    
- [Fine-tuning Mistral-7b with QLoRA on Google Colab](https://www.linkedin.com/posts/younes-belkada-b1a903145_recently-mistral-7b-has-been-released-to-activity-7117593843826339840-aPO2?utm_source=share&utm_medium=member_ios)
    
    `LLM` `Mistral-7b` `QLoRA` `Hugging Face` `TRL` `PEFT`
    
    The article describes how to fine-tune the Mistral-7b language model using QLoRA on Google Colab. This can be done using the TRL and PEFT tools from the Hugging Face ecosystem. The article also includes links to the Google Colab notebook and a GitHub thread with more information.
    
- [Instruction-tuning 101](https://twitter.com/Swarooprm7/status/1669610968165523457)
    
    `InstructGPT` `T0` `The Turking Test` `FLAN` `Natural Instructions`
    
    Instruction-tuning is a method for improving the performance of language models on a given task by providing them with additional instructions. This can be done by either fine-tuning the model on a dataset of instructions or by using a pre-trained model and providing it with instructions at inference time. Instruction-tuning has been shown to be effective for a variety of tasks, including text summarization, question answering, and machine translation.
    
- [LLM Reasoning Capabilities Improve with Increased Parameters](https://www.linkedin.com/posts/llamaindex_we-did-a-complete-survey-of-llama2-chat-7b-activity-7123113269271138304-FPoO)
    
    `reasoning` `structured outputs` `fine-tuning`
    
    A survey of llama2-chat models shows that reasoning capabilities improve as the number of parameters increases. However, structured outputs remain a challenge. This suggests that fine-tuning for better structured data extraction could potentially help.
    
- [Finetuning Overview](https://gpt-index.readthedocs.io/en/latest/end_to_end_tutorials/finetuning.html)
    
    `Finetuning` `In-context learning` `Retrieval augmentation` `Embedding finetuning` `LLM finetuning` `LlamaIndex integrations`
    
    Finetuning a model involves updating the model itself over a set of data to improve the model in various ways. This can include improving the quality of outputs, reducing hallucinations, memorizing more data holistically, and reducing latency/cost. The core of our toolkit revolves around in-context learning / retrieval augmentation, which involves using the models in inference mode and not training the models themselves. While finetuning can be also used to “augment” a model with external data, finetuning can complement retrieval augmentation in a variety of ways.
    
- [T-Few Finetuning: Efficient Training and Scalable Serving of Large Language Models](https://txt.cohere.com/tfew-finetuning/)
    
    `large language models` `finetuning` `T-Few` `training efficiency` `serving scalability`
    
    T-Few finetuning is a technique that selectively updates only a fraction of the model's weights, thus reducing training time and computational resources. It also enables model stacking, which allows for the concurrent inference of multiple finetunes, maximizing GPU utilization and improving serving scalability.
    
- [How to Fine-tune Llama 2 Embeddings for Better Retrieval Performance](https://medium.com/llamaindex-blog/fine-tuning-embeddings-for-rag-with-synthetic-data-e534409a3971)
    
    `LLM` `RAG` `Embedding Finetuning` `LlamaIndex`
    
    This article provides a step-by-step guide on how to fine-tune Llama 2 embeddings for better retrieval performance in RAG systems. The guide includes instructions on how to generate training data, fine-tune the embedding model, and evaluate the performance of the fine-tuned model.
    
- [RL4LMs: A Modular RL Library for Fine-Tuning Language Models to Human Preferences](https://github.com/allenai/RL4LMs)
    
    `language models` `reinforcement learning` `natural language processing`
    
    RL4LMs is a modular RL library for fine-tuning language models to human preferences. It provides easily customizable building blocks for training language models, including implementations of on-policy algorithms, reward functions, metrics, datasets, and LM-based actor-critic policies.
    
- [Exploring Alternatives to RLHF for Fine-Tuning Large Language Models](https://argilla.io/blog/mantisnlp-rlhf-part-1)
    
    `Large Language Models` `Supervised Fine-Tuning` `Reinforcement Learning from Human Feedback` `Direct Preference Optimization` `Chain of Hindsight`
    
    This blog explores alternatives to Reinforcement Learning from Human Feedback (RLHF) for fine-tuning large language models. The alternatives discussed include supervised fine-tuning and direct preference optimization. The blog also provides a hands-on guide to preparing human preference data and using the Transformers Reinforcement Learning library to fine-tune a large language model using direct preference optimization.
    

## 👉 Evaluating LLMs

![Evaluating LLMs](./../../images/session_1/part_3_landscape_of_llms/Large%20Language%20Models-evaluating-llms.png)

- [LMFlow Benchmark: An Automatic Evaluation Framework for Open-Source LLMs](https://optimalscale.github.io/LMFlow/blogs/benchmark.html)
    
    `LLM Evaluation` `Chatbot Arena` `GPT-4` `LMFlow Benchmark`
    
    The paper introduces LMFlow benchmark, a new benchmark which provides a cheap and easy-to-use evaluation framework that can help reflect different aspects of LLMs.
    
- [Evaluating LLM Performance](https://www.linkedin.com/posts/deshwalmahesh_nlp-llm-evaluation-activity-7123163857698643969-TBjZ)
    
    `LLM Evaluation` `RAG` `Hallucinations` `Metrics`
    
    This article discusses various techniques for evaluating LLM performance, including hallucination detection and metrics-based approaches. It also provides a framework for optimizing LLM performance using RAG and fine-tuning.
    
- [A Metrics-First Approach to LLM Evaluation](https://www.linkedin.com/posts/bhavsarpratik_llm-hallucination-activity-7112445887888457728-kdHj?utm_source=share&utm_medium=member_ios)
    
    `LLM Evaluation` `Human Evaluation` `Traditional Metrics` `Galileo Metrics`
    
    The industry has started adopting LLMs for various applications, but evaluating their performance is challenging. Human evaluation is costly and prone to errors, traditional metrics have poor correlations with human judgment, and reliable benchmarks are absent. Galileo has built metrics to help evaluate LLMs in minutes instead of days.
    
- [Evaluation Driven Development for LLM Apps](https://www.linkedin.com/posts/llamaindex_every-ai-engineer-building-llm-apps-for-prod-activity-7119774357991690242-7qnp?utm_source=share&utm_medium=member_ios)
    
    `Evaluation Driven Development` `LLM` `EDD` `Stochastic nature of LLMs` `LlamaIndex` `Retrieval methods` `Comparing LLMs`
    
    The article discusses the importance of Evaluation Driven Development (EDD) for building LLM apps. It provides a step-by-step guide to EDD, including defining evaluation metrics, defining an evaluation dataset, and trying out different approaches. The article also highlights the importance of EDD for mitigating the risks associated with the stochastic nature of LLMs. Finally, the article provides links to additional resources on EDD.
    
- [How to Evaluate Chatbots with Large Language Models](https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG)
    
    `Chatbots` `LLM` `RAG` `Evaluation` `MLflow`
    
    This article explores how to evaluate chatbots with large language models (LLMs). It discusses the use of LLMs as judges for automated evaluation, and provides best practices for using LLM judges. The article also discusses the importance of using use-case-specific benchmarks for evaluation.
    
- [How to Monitor NDCG for Ranking Models in Production](https://towardsdatascience.com/demystifying-ndcg-bee3be58cfe0#:~:text=models%20in%20production.-,What%20Is%20NDCG%20and%20Where%20Is%20It%20Used%3F,or%20other%20information%20retrieval%20system)
    
    `Ranking models` `NDCG` `ML observability` `Model monitoring` `Machine learning`
    
    This article provides a comprehensive guide to monitoring Normalized Discounted Cumulative Gain (NDCG) for ranking models in production. It covers the intuition behind NDCG, its calculation, and how it can be used to evaluate the performance of ranking models. Additionally, the article discusses the challenges of maintaining ranking models in production and how ML observability can help.
    
- [Index Metrics for Evaluating Recommender System Performance](https://towardsdatascience.com/ranking-evaluation-metrics-for-recommender-systems-263d0a66ef54)
    
    `Recommender Systems` `Evaluation Metrics` `Hit Ratio` `MRR` `Precision` `Recall` `MAP` `NDCG`
    
    Recommender systems output a ranking list of items. Hit ratio, MRR, Precision, Recall, MAP, NDCG are commonly used metrics to evaluate the performance of recommender systems.
    

## 👉 LLMs Deployment

![LLMs Deployment](./../../images/session_1/part_3_landscape_of_llms/Large%20Language%20Models-deployment.png)

- [Model Serving Frameworks for 2023](https://www.linkedin.com/posts/aboniasojasingarayar_llm-llmops-mlops-activity-7117777649896210432-IA5B?utm_source=share&utm_medium=member_ios)
    
    `Model Serving` `AI` `Machine Learning` `MLOps`
    
    The article provides a comprehensive list of model serving frameworks for AI applications in 2023. It highlights the benefits and features of each framework, including BentoML, Jina, and Torchserve, and emphasizes their importance in the MLOps process.
    
- [vLLM: A High-Throughput Library for Large Language Model Serving](https://vllm.ai/)
    
    `LLM` `machine learning` `artificial intelligence` `natural language processing`
    
    vLLM is an open-source library for fast LLM inference and serving. It utilizes PagedAttention, a new attention algorithm that effectively manages attention keys and values. vLLM equipped with PagedAttention redefines the new state of the art in LLM serving: it delivers up to 24x higher throughput than HuggingFace Transformers, without requiring any model architecture changes.
    
- [How to Optimize Latency for Open Source Language Models](https://hamel.dev/notes/llm/inference/03_inference.html)
    
    `Optimization` `Latency` `LLM` `Model Serving` `Inference`
    
    This study explores various approaches to optimizing latency for open-source LLMs. The author evaluates the effectiveness of different tools and techniques, including CTranslate2, TGI, bitsandbytes, AutoGPTQ, ExLlama, vLLM, and HuggingFace's hosted inference platform. The results show that vLLM is currently the fastest solution for distributed inference, while HuggingFace's hosted inference platform offers the best performance for single-GPU inference.
    
- [How to Optimize Large Language Model (LLM) Inference](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
    
    `Large Language Model` `LLM` `Inference` `Optimization` `Machine Learning`
    
    This article provides best practices for optimizing LLM inference, including identifying the optimization target, paying attention to the components of latency, utilizing memory bandwidth, batching, and exploring deeper systems optimizations. It also discusses hardware configurations and the importance of data-driven decisions.
    
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference)
    
    `HuggingFace` `LLM` `Rust` `Python` `gRPC` `Docker` `CUDA` `NCCL` `OpenTelemetry` `quantization`
    
    Text Generation Inference (TGI) is a toolkit for deploying and serving Large Language Models (LLMs). It implements many features such as optimized models, tensor parallelism, and distributed tracing. TGI can be installed locally or used as a Docker container.
    
- [Introducing text-embeddings-inference (TEI): A blazing fast server for sentence or document embedding](https://www.linkedin.com/posts/julienchaumond_you-already-knew-text-generation-inference-activity-7120407200228876288-cZ5U?utm_source=share&utm_medium=member_ios)
    
    `Machine Learning` `Natural Language Processing` `Text Embeddings` `Serverless Computing`
    
    TEI is a new server for sentence or document embedding that is optimized for speed and efficiency. It is based on the `candle` rust backend and does not require torch, making it very small and lightweight. TEI is a step towards real ML serverless and has the potential to make it easier to use multimodal embeddings in production.
    
- [Text Generation Inference: A Rust, Python, and gRPC toolkit](https://github.com/huggingface/text-generation-inference)
    
    `HuggingFace` `Hugging Chat` `Inference API` `Inference Endpoint` `Large Language Models (LLMs)` `Llama` `Falcon` `StarCoder` `BLOOM` `GPT-NeoX` `Open Telemetry` `Prometheus` `Tensor Parallelism` `Server-Sent Events (SSE)` `transformers.LogitsProcessor` `Custom Prompt Generation` `Fine-tuning Support`
    
    Text Generation Inference (TGI) is a toolkit for deploying and serving Large Language Models (LLMs). It supports many features such as simple launcher, production readiness, tensor parallelism, token streaming, continuous batching, optimized transformers code, quantization, watermarking, logits warper, stop sequences, log probabilities, custom prompt generation, and fine-tuning support.
    
- [LoRAX: The LLM Inference Server that Speaks for the GPUs](https://www.linkedin.com/posts/travisaddair_lora-exchange-lorax-serve-100s-of-fine-tuned-activity-7120819275442896896-vlI_)
    
    `LLM` `LoRA` `GPU` `Cloud` `Predibase`
    
    LoRAX is a new kind of LLM inference solution designed to make it cost effective and scalable to serve many fine-tuned models in production at once, conserving precious GPUs by dynamically exchanging in and out fine-tuned LoRA models within a single LLM deployment.
    

### Running LLMs Locally

- [Run Large Language Models on Your CPU with Llama.cpp](https://pub.towardsai.net/high-speed-inference-with-llama-cpp-and-vicuna-on-cpu-136d28e7887b)
    
    `LLM` `Inference` `CPU` `GPU` `ChatGPT` `Vicuna` `GPT4ALL` `Alpaca` `ggml`
    
    This article explains how to set up llama.cpp on your computer to run large language models on your CPU. It focuses on Vicuna, a chat model behaving like ChatGPT, but also shows how to run llama.cpp for other language models.
    
- [h2oGPT - 100% Private, 100% Local Chat with a GPT](https://youtu.be/Coj72EzmX20)
    
    `LLM` `h2oGPT` `Open Source` `Private` `Local`
    
    This video shows how to install and use h2oGPT, an open-source large language model (LLM), on a local computer for private, local chat with a GPT.
    
- [Run Large Language Models on Your Own Computer with llama.cpp](https://www.xzh.me/2023/09/a-perplexity-benchmark-of-llamacpp.html?m=1)
    
    `Large Language Models` `Llama.cpp` `NVIDIA CUDA` `Ubuntu 22.04`
    
    This blog post provides a step-by-step guide for running the Llama-2 7B model using llama.cpp, with NVIDIA CUDA and Ubuntu 22.04.
    
- [Get up and running with Llama 2 and other large language models locally](https://github.com/jmorganca/ollama)
    
    `LLM` `Ollama` `Modelfile` `Docker` `REST API`
    
    This article provides instructions on how to get up and running with Llama 2 and other large language models locally. It covers topics such as installing Docker, downloading models, customizing prompts, and using the REST API.
    
- [GPT4All: A Free, Local, Privacy-Aware Chatbot](https://gpt4all.io/index.html)
    
    `privacy` `local` `chatbot`
    
    GPT4All is a free-to-use, locally running chatbot that does not require a GPU or internet connection. It is designed to be privacy-aware and does not collect or store any user data.
    
- [LocalAI: An Open Source OpenAI Alternative](https://localai.io/)
    
    `LLM` `OpenAI` `gpt-3` `localai`
    
    LocalAI is a free, open-source alternative to OpenAI that allows you to run LLMs, generate images, audio, and more locally or on-prem with consumer-grade hardware. It does not require a GPU and supports multiple model families that are compatible with the ggml format.
    
- [LocalGPT: Chat with your documents on your local device using GPT models](https://github.com/PromtEngineer/localGPT)
    
    `localgpt` `gpt-3` `language-models` `privacy` `security`
    
    LocalGPT is an open-source initiative that allows you to converse with your documents without compromising your privacy. With everything running locally, you can be assured that no data ever leaves your computer.
    
- [Run any Llama 2 locally with gradio UI on GPU or CPU from anywhere (Linux/Windows/Mac)](https://github.com/liltom-eth/llama2-webui)
    
    `GPU` `CPU` `Linux` `Windows` `Mac` `Llama 2` `gradio UI` `Generative Agents/Apps`
    
    This project enables users to run any Llama 2 model locally with a gradio UI on GPU or CPU from anywhere (Linux/Windows/Mac). It uses `llama2-wrapper` as the local llama2 backend for Generative Agents/Apps.
    

### Semantic Cache for LLMs

- [GPTCache: Semantic Cache for LLMs](https://github.com/zilliztech/GPTCache/tree/main)
    
    `LLM` `Semantic Caching` `LangChain` `Llama Index`
    
    GPTCache is a semantic cache for LLMs that helps reduce the cost and latency of LLM API calls. It uses embedding algorithms to convert queries into embeddings and uses a vector store for similarity search on these embeddings. This allows GPTCache to identify and retrieve similar or related queries from the cache storage, thereby increasing cache hit probability and enhancing overall caching efficiency.


## 👉 LLMs Inference Optimisation

![LLMs Inference Optimisation](./../../images/session_1/part_3_landscape_of_llms/Large%20Language%20Models-quantisation.png)

### LLM Quantization

- **[BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)**
    
    `Transformers` `Quantization` `LLM`
    
    BitNet is a scalable and stable 1-bit Transformer architecture designed for large language models. It achieves competitive performance while substantially reducing memory footprint and energy consumption, compared to state-of-the-art 8-bit quantization methods and FP16 Transformer baselines. BitNet exhibits a scaling law akin to full-precision Transformers, suggesting its potential for effective scaling to even larger language models while maintaining efficiency and performance benefits.
    
- [HuggingFace: An Overview of Natively Supported Quantization Schemes in Transformers](https://www.linkedin.com/posts/prithivirajdamodaran_google-colaboratory-activity-7112035540165619712-fEzF?utm_source=share&utm_medium=member_ios)
    
    `HuggingFace` `Transformers` `Quantization`
    
    The article provides an overview of natively supported quantization schemes in Transformers, including bitsandbytes and GPTQ. It also discusses the relation between bitsandbytes and GPTQ, and compares the performance of GPTQ with bitsandbytes nf4.
    
- [Hugging Face Optimum GPTQ Quantization](https://www.philschmid.de/gptq-llama)
    
    `Hugging Face` `Optimum` `GPTQ` `Quantization` `LLM` `NLP`
    
    This blog post introduces GPTQ quantization, a method to compress GPT models by reducing the number of bits needed to store each weight. It also provides a step-by-step tutorial on how to quantize a GPT model using the Hugging Face Optimum library.
    
- [SqueezeLLM: Efficient LLM Serving with Dense-and-Sparse Quantization](https://github.com/SqueezeAILab/SqueezeLLM)
    
    `Model Compression` `Quantization` `Efficient Serving`
    
    SqueezeLLM is a post-training quantization framework that incorporates a new method called Dense-and-Sparse Quantization to enable efficient LLM serving. This method splits weight matrices into two components: a dense component that can be heavily quantized without affecting model performance, and a sparse part that preserves sensitive and outlier parts of the weight matrices. With this approach, SqueezeLLM is able to serve larger models with smaller memory footprint, the same latency, and yet higher accuracy and quality.
    
- [SqueezeLLM: Achieving 3-bit Quantization for LLM Inference Acceleration](https://www.linkedin.com/posts/sanyambhutani_3-bit-quantisation-is-here-squeezellm-activity-7076906092802244608-8frQ?utm_source=share&utm_medium=member_ios)
    
    `Post-Training Quantisation (PQT)` `Non-Uniform Quantization` `Dense and Sparse Quantization` `Memory Bottlenecked Operations` `GPU Memory Optimization` `Model Compression` `LLM Inference Acceleration`
    
    The paper proposes SqueezeLLM, a novel Post-Training Quantisation (PQT) technique that achieves 3-bit quantization for LLM inference acceleration. It introduces non-uniform quantization and dense and sparse quantization to address memory bottlenecks and achieve 230% speedup in inference. The paper also compares SqueezeLLM with other quantization techniques and demonstrates its superior performance in terms of compression and accuracy.
    
- [New Research Paper: Sparse Quantized Representation for Efficient Large Language Model Compression](https://www.linkedin.com/posts/karan-malhotra-44864b10a_230603078pdf-activity-7076079427431923712-NP9d?utm_source=share&utm_medium=member_ios)
    
    `LLM Compression` `SpQR` `Quantization` `Falcon` `LLaMA`
    
    A new research paper introduces Sparse Quantized Representation (SpQR), a new compression format and quantization technique that enables near-lossless compression of LLMs down to 3-4 bits per parameter. This technique works by recognizing and isolating outlier weights that cause large quantization errors, and storing them in higher precision, while compressing all other weights to 3-4 bits. The authors claim that SpQR can achieve relative accuracy losses of less than 1% in perplexity for highly accurate LLMs like Falcon and LLaMA.
    
- [Two Cool Releases from Last Week in the LLM Domain](https://www.linkedin.com/posts/sanyambhutani_two-really-cool-releases-from-last-week-activity-7073834050515329024-qysu?utm_source=share&utm_medium=member_ios)
    
    `RedPajama Dataset` `LLM Model Family` `HELM Benchmark`
    
    Cerebras Systems has released a cleaned and de-duplicated version of the RedPajama Dataset, reducing its size by 49%. Additionally, RedPajama has released a model family of 7B size, including chat, instruction fine-tuned, and base models. The instruction fine-tuned model shows promising performance on the HELM benchmark.
    
## 👉 LLMs with Large Context Window

![LLMs with Large Context Window](./../../images/session_1/part_3_landscape_of_llms/Large%20Language%20Models-large-context.png)

- [How to use 100K context window in LLMs](https://blog.gopenai.com/how-to-speed-up-llms-and-use-100k-context-window-all-tricks-in-one-place-ffd40577b4c?gi=01371942e829)
    
    `LLM Training` `Model Size` `Attention Mechanisms`
    
    This article explores techniques to speed up training and inference of LLMs to use large context window up to 100K input tokens. It covers ALiBi positional embedding, Sparse Attention, FlashAttention, Multi-Query attention, Conditional computation, and the use of 80GB A100 GPUs.
    
- [XGen: A New State-of-the-Art 7B LLM with Standard Dense Attention on Up to 8K Sequence Length](https://www.linkedin.com/posts/caiming-xiong-150a1417_gpt-largelanguagemodels-nlp-activity-7079888057126047744-jftG?utm_source=share&utm_medium=member_ios)
    
    `LLM` `NLP` `Machine Learning` `Artificial Intelligence`
    
    XGen is a new state-of-the-art 7B LLM with standard dense attention on up to 8K sequence length. It achieves comparable or better results than other open-source LLMs of similar model size on standard NLP benchmarks. XGen also shows benefits on long sequence modeling benchmarks and achieves great results on both text and code tasks.
    

## 👉 Challanges with LLMs

![Challanges with LLMs](./../../images/session_1/part_3_landscape_of_llms/Large%20Language%20Models-challanges.png)

- [Challenges in Building LLM Applications for Production](https://home.mlops.community/public/videos/building-llm-applications-for-production?utm_campaign=LLM%20II%20%231&utm_content=LLM%20in%20production%20keynotes%20are%20out%21&utm_medium=email&utm_source=ActiveCampaign)
    
    `Consistency` `Hallucinations` `Privacy` `Context Length` `Data Drift` `Model Updates and Compatibility` `LM on the Edge` `Model Size` `Non-English Languages` `Chat vs. Search as an Interface` `Data Bottleneck` `Hype Cycles and the Importance of Data`
    
    This talk discusses the challenges in building LLM applications for production. These challenges include consistency, hallucinations, privacy, context length, data drift, model updates and compatibility, LM on the edge, model size, non-English languages, chat vs. search as an interface, data bottleneck, and hype cycles and the importance of data.
    
- [Open challenges in LLM research](https://huyenchip.com/2023/08/16/llm-research-open-challenges.html)
    
    `hallucinations` `context learning` `multimodality` `new architecture` `GPU alternatives` `agent usability` `learning from human preference` `chat interface efficiency` `non-English language support`
    
    The article discusses the ten major research directions in the field of LLMs, including reducing and measuring hallucinations, optimizing context length and construction, incorporating other data modalities, making LLMs faster and cheaper, designing new model architectures, developing GPU alternatives, making agents usable, improving learning from human preference, improving the efficiency of the chat interface, and building LLMs for non-English languages.
    
- [The Perils of Blindly Reusing Pre-trained Language Models](https://www.linkedin.com/posts/prithivirajdamodaran_dont-blindly-reuse-pre-trained-models-from-activity-7082591506200457217-hxEm?utm_source=share&utm_medium=member_ios)
    
    `NLP` `Transfer Learning` `Model Analysis` `WeightWatchers`
    
    Reusing pre-trained language models without careful consideration can lead to negative impacts on downstream tasks due to issues such as over-training, under-training, or over-parameterization. WeightWatchers is an open-source diagnostic tool that can be used to analyze DNNs without access to training or test data, helping to identify potential issues before deployment.
    

### Large vs Small Langage Models

- [Small language models can outperform LLMs in specific domains](https://www.linkedin.com/posts/sebastien-bubeck-6b558a1a5_textbooks-are-all-you-need-activity-7077091292077330433-w2vZ?utm_source=share&utm_medium=member_ios)
    
    `LLM` `NLP` `Machine Learning`
    
    A new LLM trained by Microsoft Research achieves 51% on HumanEval with only 1.3B parameters and 7B tokens training dataset, outperforming much larger LLMs. This suggests that smaller language models can be more effective in specific domains, such as Python code-generation.
    
- [Are Large Language Models All We Need?](https://www.linkedin.com/posts/lijiali_paper-page-textbooks-are-all-you-need-activity-7078523739017003008-e-jK?utm_source=share&utm_medium=member_ios)
    
    `LLM` `Model Size` `Data Quality`
    
    The author discusses the recent trend of focusing on model sizes in the field of LLMs and argues that data quality is often overlooked. They cite the example of phi-1, a 1.3B parameter Transformer-based model by Microsoft, which achieved surprisingly good results. The author concludes that we should pay more attention to data quality when developing LLMs.


## 👉 LLM Applications

![LLM Applications](./../../images/session_1/part_3_landscape_of_llms/Large%20Language%20Models-applications.png)    


### LLMs for Translation

- [ParroT: Enhancing and Regulating Translation Abilities in Chatbots with Open-Source LLMs](https://www.linkedin.com/posts/ricky-costa-nlp_github-wxjiaoparrot-the-parrot-framework-activity-7095024621321662464-5HWn?utm_source=share&utm_medium=member_ios)
    
    `LLM` `Translation` `Chatbots` `ParroT`
    
    The ParroT framework enhances and regulates the translation abilities of chatbots by leveraging open-source LLMs and human-written translation and evaluation data.
    

### LLMs For Mobile App Developers

- [Hugging Face releases tools for Swift developers to incorporate language models in their apps](https://www.linkedin.com/posts/sahar-mor_a-big-step-forward-for-on-device-llms-activity-7095782887811125248-H5f2?utm_source=share&utm_medium=member_ios)
    
    `Hugging Face` `Swift` `transformers` `Core ML` `Llama` `Falcon`
    
    Hugging Face has released a package and tools to help Swift developers incorporate language models in their apps, including swift-transformers, swift-chat, transformers-to-coreml, and ready-to-use LLMs such as Llama 2 7B and Falcon 7B.
    

### LLM Assistants

- [Comparing coding assistants](https://www.linkedin.com/posts/kalyanksnlp_llms-opensource-nlproc-activity-7077128568891207680-umhc?utm_source=share&utm_medium=member_ios)
    
    `Rust` `coding assistants` `best practices`
    
    The author asks for advice on how to compare coding assistants. They are concerned about using an assistant for Rust because they are not savvy enough to catch certain bugs. Kalyan KS suggests that the author try out Falcoder, a coding assistant that uses the Falcon-7B model and instruction tuning.
    
- [GPT-Engineer: An AI Agent That Can Write Entire Codebases](https://www.linkedin.com/posts/ugcPost-7076254210232643584-JoY8?utm_source=share&utm_medium=member_ios)
    
    `Artificial Intelligence` `Machine Learning` `Natural Language Processing` `Programming`
    
    GPT-Engineer is an AI agent that can write entire codebases with a prompt and learn how you want your code to look. It asks clarifying questions, generates technical specifications, writes all necessary code, and lets you easily add your own reasoning steps, modify, and experiment. With GPT-Engineer, you can finish a coding project in minutes.
    
- [Introducing AssistGPT: A General Multi-modal Assistant](https://www.linkedin.com/posts/aleksagordic_daily-paper-time-assistgpt-a-general-activity-7075519171983228928-7Ltf?utm_source=share&utm_medium=member_ios)
    
    `Multimodality` `Language and Code` `ReAct Agent` `Planning and Execution`
    
    The paper introduces AssistGPT, a general multi-modal assistant that can plan, execute, inspect, and learn. It combines many of the latest trends in AI, including multimodality, language and code, and ReAct agents. The paper also includes a cool demo and discusses the latency of the system.
    
- [GPTeam: Building Human-like Social Behavior in Language Models](https://blog.langchain.dev/gpteam-a-multi-agent-simulation/)
    
    `Multi-agent simulation` `Human-like social behavior` `Language models` `Generative agents`
    
    GPTeam is a completely customizable open-source multi-agent simulation, inspired by Stanford’s ground-breaking “Generative Agents” paper. Every agent within a GPTeam simulation has their own unique personality, memories, and directives, leading to interesting emergent behavior as they interact.
    
- [LLM Powered Autonomous Agents](https://www.linkedin.com/posts/sanyambhutani_llm-powered-autonomous-agents-lilian-activity-7079812038943809536-0xsL?utm_source=share&utm_medium=member_ios)
    
    `Large Language Model` `Planning` `Memory` `Tool Use`
    
    The article provides a comprehensive overview of building Large Language Model powered agents, including relevant papers, practical applications, and case studies.
    
- [Best write-up ever on LLM Agents](https://www.linkedin.com/posts/prithivirajdamodaran_best-write-ups-ever-on-llm-agents-by-my-favourite-activity-7079821641832091648-hM-l?utm_source=share&utm_medium=member_ios)
    
    `LLM` `NLP` `OpenAI`
    
    The article provides a comprehensive overview of LLM agents, including their capabilities, limitations, and potential applications. It also discusses the challenges involved in developing and deploying LLM agents, and the ethical considerations that need to be taken into account.
    

### Retrieval Augmented Generation

- [RAG & Enterprise: A Match Made in Heaven](https://www.linkedin.com/posts/prithivirajdamodaran_usecase-retrieval-augmented-generation-for-activity-7076430925122691072-nYMj?utm_source=share&utm_medium=member_ios)
    
    `RAG` `LLM` `Enterprise Search` `Information Retrieval`
    
    RAG (Retrieve and Generate) models are a powerful tool for enterprise search, as they offer flexibility, practicality, broader coverage, and interpretability. Additionally, with the help of tools like LangChain and Google Vertex, it is now easier than ever to implement RAG solutions.
    
- [HNSW-FINGER: Approximate Nearest Neighbor Search with Locality-Sensitive Hashing](https://youtu.be/OsxZG2XfcZA)
    
    `locality-sensitive hashing` `approximate nearest neighbor search` `HNSW`
    
    HNSW-FINGER is a new approximate nearest neighbor search algorithm that uses locality-sensitive hashing to project the query and candidate nodes onto a center node. This allows HNSW-FINGER to achieve better accuracy and efficiency than existing approximate nearest neighbor search algorithms.
    
- [Vector Databases and Hierarchical Navigable Small World](https://www.linkedin.com/posts/damienbenveniste_machinelearning-datascience-artificialintelligence-activity-7085279691611213824-RnrZ?utm_source=share&utm_medium=member_ios)
    
    `Vector Databases` `Machine Learning` `Artificial Intelligence` `Data Science` `Generative AI`
    
    The article discusses the rise of vector databases in the era of generative AI and introduces Hierarchical Navigable Small World (HNSW) as an efficient indexing method. HNSW builds multiple graph layers with varying densities to optimize the search process and reduce the number of iterations required to find approximate nearest neighbors.
    
- [RAG-Fusion: A New Retrieval Technique for LLM](https://www.linkedin.com/posts/langchain_rag-fusion-a-new-retrieval-technique-activity-7122978999038857216-xTB8)
    
    `LLM` `Retrieval` `MultiQueryRetrieval` `Reciprocal Rank Fusion`
    
    RAG-Fusion is a new retrieval technique that builds upon the idea of MultiQueryRetrieval. It generates multiple sub queries based on a user question, retrieves documents for each sub query, and merges the retrieved documents together using Reciprocal Rank Fusion.
    
- [Question Answering over Documents with Retrieval-Augmented Generation](https://python.langchain.com/docs/use_cases/question_answering.html)
    
    `rag` `question answering` `information retrieval` `llm`
    
    This article describes how to build a question-answering over documents application using LLMs. The article covers the use of retrieval-augmented generation (RAG) for this task, and provides a walkthrough of how to build such an application.
    
- [Reordering retrieved documents to improve performance](https://python.langchain.com/docs/modules/data_connection/document_transformers/post_retrieval/long_context_reorder)
    
    `long context` `performance degradation` `retrieval`
    
    When models must access relevant information in the middle of long contexts, they tend to ignore the provided documents. This issue can be avoided by reordering documents after retrieval to avoid performance degradation.
    
- [How to improve the performance of your LLM search engine with Retrieve & Re-Rank](https://www.sbert.net/examples/applications/retrieve_rerank/README.html)
    
    `LLM` `Semantic Search` `Information Retrieval` `Question Answering`
    
    This article explains how to improve the performance of your LLM search engine with Retrieve & Re-Rank. It covers lexical search, dense retrieval, semantic search and cross-encoders.
    
- [EAR: Improving Passage Retrieval for Open-Domain Question Answering with Query Expansion and Reranking](https://virtual2023.aclweb.org/paper_P452.html?utm_source=linkedin&utm_medium=organic_social&utm_campaign=acl2023&utm_content=image)
    
    `passage retrieval` `query expansion` `query reranking` `open-domain question answering`
    
    EAR is a query Expansion And Reranking approach for improving passage retrieval, with the application to open-domain question answering. EAR first applies a query expansion model to generate a diverse set of queries, and then uses a query reranker to select the ones that could lead to better retrieval results.
    
- [MS MARCO: A Large Scale Information Retrieval Corpus](https://www.sbert.net/docs/pretrained-models/msmarco-v3.html)
    
    `information retrieval` `semantic search` `TREC-DL 2019` `MS Marco Passage Retrieval` `BM25` `ElasticSearch` `electra-base-model` `cross-encoder`
    
    MS MARCO is a large scale information retrieval corpus that was created based on real user search queries using Bing search engine. It can be used for semantic search, i.e., given keywords / a search phrase / a question, the model will find passages that are relevant for the search query. Performance is evaluated on TREC-DL 2019 and MS Marco Passage Retrieval dataset. As baseline we show the results for lexical search with BM25 using ElasticSearch.
    
- [Self-querying retriever: A new way to search for information](https://python.langchain.com/docs/modules/data_connection/retrievers/self_query)
    
    `LLM` `VectorStore` `information retrieval`
    
    A self-querying retriever is a new way to search for information that uses a query-constructing LLM chain to write a structured query and then applies that structured query to its underlying VectorStore. This allows the retriever to not only use the user-input query for semantic similarity comparison with the contents of stored documents but to also extract filters from the user query on the metadata of stored documents and to execute those filters.
    
- [GenQ: Training Effective Dense Retrieval Models with Synthetic Queries](https://www.pinecone.io/learn/series/nlp/genq/)
    
    `dense retrieval` `bi-encoders` `sentence transformers` `text generation` `synthetic data` `asymmetric semantic search` `query generation` `T5` `MNR loss` `Pinecone`
    
    GenQ is a method for training effective dense retrieval models using synthetic queries. It uses a text generation model to generate queries for unlabeled passages of text, which are then used to fine-tune a bi-encoder model. GenQ can achieve performances approaching models trained with supervised methods, and it is particularly useful when we have limited labeled data.
    
- [InPars-v2: Efficient Dataset Generation for Information Retrieval with Open-Source Language Models](https://arxiv.org/abs/2301.01820)
    
    `information retrieval` `large language models` `dataset generation` `open-source`
    
    InPars-v2 is a dataset generator that uses open-source LLMs and existing powerful rerankers to select synthetic query-document pairs for training. It achieves new state-of-the-art results on the BEIR benchmark.
    
- [Qdrant: A Vector Database & Vector Similarity Search Engine](https://qdrant.tech/)
    
    `Vector database` `Vector similarity search` `Approximate nearest neighbor search` `Machine learning` `Artificial intelligence`
    
    Qdrant is a vector database and vector similarity search engine that can be used for building applications such as matching, searching, and recommending. It is easy to use and provides a variety of features such as support for additional payload associated with vectors, payload filtering conditions, and dynamic query planning.
    
- [AutoMergingRetriever: A New Algorithm for Better Retrieval and RAG](https://www.linkedin.com/posts/llamaindex_we-present-a-cool-new-algorithm-for-better-activity-7101703776104845312-8Kc9?utm_source=share&utm_medium=member_ios)
    
    `LLM` `Retrieval` `RAG` `ChatGPT` `Dynamic Retrieval` `Semantic Relatedness`
    
    The AutoMergingRetriever algorithm dynamically retrieves less disparate / larger contiguous blobs of context *only when you need it*. This helps the LLM synthesize better results, but avoids always cramming in as much context as you can.
    
- [Optimizing RAG With LLMS: Exploring Chunking Techniques and Reranking for Enhanced Results](https://www.youtube.com/watch?v=QpRTdZDR4tE)
    
    `LLM` `Chunking` `Ranking` `Retrieval Augmented Generation (RAG)`
    
    This article explores chunking techniques and reranking for enhanced results in the context of optimizing RAG with LLMs. The key points covered include strategies for optimizing RAG, using chunking techniques to streamline processing, and implementing ranking models to enhance search quality.
    
- [Dynamic chunk length in AutoMergingRetriever](https://x.com/clusteredbytes/status/1707864519433736305?s=20)
    
    `language-models` `retrieval` `summarization`
    
    The AutoMergingRetriever dynamically chooses the chunk length when retrieving information, resulting in better semantic meaning and context.
    
- [Multi-Document Agents for Building LLM-Powered QA Systems](https://www.linkedin.com/posts/llamaindex_building-good-rag-systems-is-hard-but-building-activity-7114307512664825856-4swa?utm_source=share&utm_medium=member_ios)
    
    `RAG` `LLM` `QA` `summarization` `multi-document agents`
    
    The article introduces a new approach for building LLM-powered QA systems that can scale to large numbers of documents and question types. The approach uses multi-document agents, which are able to answer a broad set of questions, including fact-based QA over single documents, summarization over single documents, fact-based comparisons over multiple documents, and holistic comparisons across multiple documents.
    
- [How to Improve Your RAG App: Adjusting Chunk Size](https://www.linkedin.com/posts/llamaindex_adjusting-your-chunk-size-is-one-of-the-first-activity-7116163608417337344-ejfk?utm_source=share&utm_medium=member_ios)
    
    `RAG` `chunk size` `retrieval` `ranking` `evaluation`
    
    Adjusting chunk size is an important step in improving the performance of a RAG app. More chunks do not always lead to better results, and reranking retrieved chunks may not necessarily improve results either. To find the optimal chunk size, it is necessary to define an evaluation benchmark and perform a sweep over chunk sizes and top-k values. The Arize AI team has provided a comprehensive Colab notebook and slides that demonstrate how to run chunk size sweeps and perform retrieval and Q&A evaluations with Phoenix and LlamaIndex.
    
- [How to Choose the Right Chunk Size for Your RAG System](https://blog.llamaindex.ai/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5)
    
    `RAG system` `chunk size` `response time` `faithfulness` `relevancy`
    
    Choosing the right chunk size for a RAG system is critical for efficiency and accuracy. The optimal chunk size strikes a balance between capturing essential information and speed. The article provides a practical evaluation setup to determine the right chunk size for a specific use case and dataset.
    
- [RAG-Fusion: The Next Frontier of Search Technology](https://github.com/Raudaschl/rag-fusion)
    
    `Reciprocal Rank Fusion` `Query Generation` `Retrieval Augmented Generation` `Vector Search`
    
    RAG-Fusion is a search methodology that aims to bridge the gap between traditional search paradigms and the multifaceted dimensions of human queries. It employs multiple query generation and Reciprocal Rank Fusion to re-rank search results, with the goal of unearthing transformative knowledge that often remains hidden behind top search results.
    
- [RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation](https://arxiv.org/abs/2310.04408)
    
    `language-models` `retrieval-augmentation` `compression` `abstractive-summarization` `extractive-summarization`
    
    We propose a method to improve the performance of retrieval-augmented language models (LMs) by compressing the retrieved documents into textual summaries. Our method, RECOMP, achieves a compression rate of as low as 6% with minimal loss in performance for both language modeling and open domain question answering tasks. We also show that our compressors trained for one LM can transfer to other LMs on the language modeling task and provide summaries largely faithful to the retrieved documents.
    
- [Optimizing Retrieval and Generation Performance in Large Language Models](https://www.linkedin.com/feed/update/urn:li:activity:7113422147821211648?utm_source=share&utm_medium=member_ios)
    
    `RAG` `Machine Learning` `Knowledge Retrieval` `AI`
    
    This article discusses various techniques for optimizing retrieval and generation performance in large language models, including decoupling chunks for retrieval and synthesis, using structured retrieval techniques, dynamically retrieving chunks based on tasks, and optimizing context embeddings.
    
- [Scaling Retrieval-Augmented LLM to 48B](https://www.linkedin.com/posts/omarsar_instruction-tuning-the-largest-pretrained-activity-7118231963194253312-6yjm?utm_source=share&utm_medium=member_ios)
    
    `LLM Scaling` `Retrieval-Augmented LLM` `Instruction Tuning`
    
    NVIDIA introduces Retro 48B, the largest LLM pretrained with retrieval. It shows significant perplexity improvement over GPT 43B and can be instruction-tuned more effectively, achieving +7% improvement on zero-shot question-answering tasks.
    
- [Parsing complex documents with embedded tables using unstructured.io and LlamaIndex](https://www.linkedin.com/posts/llamaindex_how-do-you-easily-parse-and-perform-rag-over-activity-7120853678915289088-iCPF?utm_source=share&utm_medium=member_ios)
    
    `unstructured.io` `LlamaIndex` `SEC filings` `research papers` `invoices`
    
    Parsing complex documents with embedded tables can be done using [unstructured.io](http://unstructured.io/) and LlamaIndex. This is especially relevant for SEC filings, research papers, invoices, and more.
    
- [LLM Production Ready RAGs](https://docs.google.com/presentation/d/1v7T6ejrSo87ndGeGC7tt6zeq-cftu03WWw7WL8Jskug/mobilepresent?slide=id.g2476298ff5a_0_160)
    
    `LLM` `RAG` `Best Practices`
    
    This talk will discuss best practices for creating production ready RAGs in the context of LLMs.
    
- [Joint Tabular/Semantic QA over Tesla 10K](https://docs.llamaindex.ai/en/stable/examples/query_engine/sec_tables/tesla_10q_table.html)
    
    `LLM` `NLP` `Information Retrieval` `Question Answering`
    
    This article demonstrates how to ask questions over Tesla's 10K report with understanding of both the unstructured text as well as embedded tables. It utilizes Unstructured to parse out the tables and LlamaIndex recursive retrieval to index and retrieve tables if necessary given the user question.
    
- [New Fine-Tuning Features in LlamaIndex](https://www.linkedin.com/posts/llamaindex_we-added-a-lot-of-new-fine-tuning-features-activity-7116229026754543616-i-Ab?utm_source=share&utm_medium=member_ios)
    
    `fine-tuning` `retrieval augmentation` `structured outputs`
    
    This week, LlamaIndex added a lot of new fine-tuning features, including fine-tuning with retrieval augmentation and fine-tuning for better structured outputs.
    
- [SuperKnowa: Building Reliable RAG Pipelines for Enterprise LLM Applications](https://medium.com/towards-generative-ai/superknowa-simplest-framework-yet-to-swiftly-build-enterprise-rag-solutions-at-scale-ca90b49be28a)
    
    `RAG` `LLM` `NLP` `Generative AI` `Enterprise AI`
    
    This article introduces SuperKnowa, a framework for building reliable and scalable RAG pipelines for enterprise LLM applications. It discusses the challenges of taking a RAG PoC to production and how SuperKnowa addresses these challenges. The article also provides an overview of the SuperKnowa framework and its features, including data indexing, context-aware queries, model evaluation, and debugging.
    
- [SEC Insights: A real-world full-stack application using LlamaIndex](https://github.com/run-llama/sec-insights)
    
    `LLM` `RAG` `SEC Insights` `Tutorial` `Open Source`
    
    This repository contains the code for SEC Insights, a real-world full-stack application that uses the Retrieval Augmented Generation (RAG) capabilities of LlamaIndex to answer questions about SEC 10-K & 10-Q documents. The application is open source and available on GitHub. A tutorial video is also available on YouTube.
    
- [Text Ranking with Pretrained Transformers](https://arxiv.org/abs/2010.06467)
    
    `Text Ranking` `Transformers` `BERT` `Self-supervised Learning` `Natural Language Processing` `Information Retrieval`
    
    This survey provides an overview of text ranking with neural network architectures known as transformers. We cover a wide range of modern techniques, grouped into two high-level categories: transformer models that perform reranking in multi-stage architectures and dense retrieval techniques that perform ranking directly.
    
- [8 Key Considerations for Building Production-Grade LLM Apps](https://twitter.com/jerryjliu0/status/1692931028963221929?s=20)
    
    `LLM` `RAG` `Embeddings` `Data Pipelines` `Scalability` `Retrieval` `Entity Lookup`
    
    This article discusses 8 key considerations for building production-grade LLM apps over your data. These considerations include using different chunks for retrieval and synthesis, using embeddings that live in a different latent space than the raw text, dynamically loading/updating the data, designing the pipeline for scalability, storing data in a hierarchical fashion, using robust data pipelines, and using hybrid search for entity lookup.
    

### Embeddings for Retrival

- [How to use Aleph Alpha's semantic embeddings](https://python.langchain.com/docs/modules/data_connection/text_embedding/integrations/aleph_alpha)
    
    `embeddings` `semantic embeddings` `Aleph Alpha`
    
    There are two ways to use Aleph Alpha's semantic embeddings: asymmetric embeddings and symmetric embeddings.
    
- [TaylorAI/gte-tiny: A 45MB Tiny Model That Beats Existing Sentence-Transformer Embeddings](https://www.linkedin.com/posts/prithivirajdamodaran_%3F%3F%3F%3F-%3F%3F%3F%3F-%3F%3F%3F%3F%3F-%3F%3F%3F%3F-activity-7120279840569597952-iwc-?utm_source=share&utm_medium=member_ios)
    
    `Vector Search` `Sentence Transformer` `Embedding` `VectorDB` `MTEB Leaderboard`
    
    The paper introduces TaylorAI/gte-tiny, a 45MB tiny model that beats existing sentence-transformer embedders. The model is based on BERT and distilled from thenlper/gte-small. It achieves comparable performance to larger models while being much smaller and faster. The model ranks 28th out of 126 models on the MTEB leaderboard.
    
- [LLM-based Sentence Embeddings](https://python.langchain.com/docs/modules/data_connection/text_embedding/integrations/sentence_transformers)
    
    `LLM` `Sentence Embeddings` `HuggingFaceEmbeddings` `SentenceTransformerEmbeddings` `Sentence-BERT`
    
    This article introduces a new way to generate sentence embeddings using LLM. The method is based on the HuggingFaceEmbeddings integration, which allows users to use SentenceTransformers embeddings directly. The article also provides an example of how to use the new method.
    

### Evaluating RAGs

- [Ragas: A Framework for Evaluating Retrieval Augmented Generation Pipelines](https://github.com/explodinggradients/ragas)
    
    `LLM` `RAG` `NLP` `Machine Learning`
    
    Ragas is a framework that helps you evaluate your Retrieval Augmented Generation (RAG) pipelines. It provides you with the tools based on the latest research for evaluating LLM-generated text to give you insights about your RAG pipeline. Ragas can be integrated with your CI/CD to provide continuous checks to ensure performance.
    

### Integrating LLMs with Knowledge Graphs

- [LLMs and Knowledge Graphs](https://www.linkedin.com/posts/llamaindex_google-colaboratory-activity-7101315253833011200-zEOX?utm_source=share&utm_medium=member_ios)
    
    `Knowledge Graphs` `LLMs` `RAG` `Vector Databases` `ChromaDB`
    
    This article discusses the advantages and disadvantages of using Knowledge Graphs (KGs) with LLMs. It also provides a link to a Colab notebook and a video tutorial on the topic.
    

### LLM Watermarking

- [AI generated text? New research shows watermark removal is harder than one thinks!](https://www.linkedin.com/posts/srijankr_ai-activity-7076757082720403456-eVjm?utm_source=share&utm_medium=member_ios)
    
    `LLM` `Watermarking` `Text Generation` `AI Ethics`
    
    Researchers from the University of Maryland have found that it is much harder to remove watermarks from AI-generated text than previously thought. This has implications for the use of watermarks to detect machine-generated content, such as spam and harmful content.
    
## 👉 LLM Application Development

![LLM Application Development](./../../images/session_1/part_3_landscape_of_llms/Large%20Language%20Models-app-development.png)


### LLM SaaS Apps

- [Introducing LLM Studio: A Powerful Platform for Building and Deploying Language Models](https://www.rungalileo.io/blog/announcing-llm-studio)
    
    `LLM` `NLP` `Machine Learning`
    
    LLM Studio is a powerful platform that enables developers to easily build, train, and deploy language models. With its user-friendly interface and comprehensive set of features, LLM Studio makes it easy to create and deploy state-of-the-art language models for a variety of applications.
    
- [Verba: The Open-Source LLM-Based Search Engine](https://www.linkedin.com/posts/edwardschmuhl_machinelearning-ai-llm-activity-7112466389528891393-VEtf?utm_source=share&utm_medium=member_ios)
    
    `LLM` `Open Source` `Search Engine`
    
    Verba is an open-source LLM-based search engine that supports a broad spectrum of open-source libraries and custom features. It is easy to install and use, and it does not require users to give away any of their data.
    

## 👉 LLM Courses

![LLM Courses](./../../images/session_1/part_3_landscape_of_llms/Large%20Language%20Models-courses.png)

- [LLM course recommendations](https://www.linkedin.com/posts/manishsgupta_%3F%3F%3F%3F%3F-%3F%3F%3F%3F%3F%3F-%3F%3F-%3F%3F%3F%3F%3F%3F-activity-7085475843833016320-rvZ9?utm_source=share&utm_medium=member_ios)
    
    `LLM` `NLP` `AI`
    
    The article recommends some short courses on LLM. The author also recommends some YouTube channels and videos on LLM.