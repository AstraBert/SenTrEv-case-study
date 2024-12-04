# Evaluation of the performance of three _Sentence Transformers_ text embedders - a case study for SenTrEv

Astra Clelia Bertelli[^1]

**Abstract**: As of december 2024, more than 10,000 models are available with the _sentence-transformers_ library: with the growing number of Retrieval Augmented Generation applications, it is more than necessary to employ an easy-to-use evaluation method to find the best text embedder for retrieval purposes. Sentence Transformers Evaluator (SenTrEv) is a python package that was developed to provide user with a simple evaluation of Sentence Transformers text embedding models on PDF data. In this case study, we present the potential application of SenTrEv in the evaluation of three popular _sentence-transformers_ text embedders, reporting and interpreting the statistics related to retrieval success rate, time performance, mean reciprocal ranking and carbon emissions.

## Introduction

The growth of Artificial Intelligence (AI) from late 2022 has benefited numerous fields, ranging from chat models to computer vision to the generation of audiovisual material.

A field that has been growing, since AI-based text generation has been given special attention from the generalist public, is the one of Retrieval Augmented Generation (RAG). The idea behind RAG is simple but powerful: 

- it starts from some text-based material (being it PDFs, CSVs, HTML websites, Markdown documents...) 
- the text is chunked, i.e. subdivided into smaller pieces to make it more "digestible"
- a model (called text embedder, encoder or vectorizer, and we will use those expressions interchangeably in this study) takes care of generating a multi-dimensional vector representation of the text
- the vector representations are loaded into a vector database (there are many popular services that offer this kind of data storage)
- when the chat model is prompted with a query, this same query is firstly embedded into a vector and then used to retrieve similar chunks of texts from the vector database: these chunks of text will serve as context that may enable the model to produce an informed answer to the user

This approach, although sometimes much more complicated than the core here reported, has proven to substantially enhance AI models performance.

Nevertheless, RAG comes with an important hassle: one cannot assess the goodness of a text embedder until it tests it in action, which could lead to two main consequences - either the model is too small for the pipeline to which it is applied, resulting in low-quality embeddings and retrieval calls, or the model is too big, ending up in a waste of computational and (potentially) financial resources. It can thus be concluded that an evaluation framework for embedding models should be employed to establish their fitness for the task before they are employed in any test or production environment.

A potential solution to evaluate the retrieval performance of models is SenTrEv (**Sen**tence **Tr**ansformers **Ev**aluator; Bertelli, 2024).

SenTrEv is a Python package offering a simple retrieval evaluation frameworks that takes into account retrieval accuracy, time performance and carbon emission tracking. Its applicability concerns PDF data and extends to all 10,000+ embedders available through the popular `sentence-transformers` Python library (https://sbert.net, last visited 12-04-2024), leveraging `Qdrant` (https://qdrant.tech, last visited 12-04-2024) as vector storage.

In this case study, we showcase SenTrEv evaluation potential by comparing three popular `sentence-transformers` based embedding models on a small PDF dataset.

## Materials and Methods

### Embedding models

As already mentioned in the introduction, SenTrEv was employed in the evaluation process of three popular `sentence-transformers` encoders: `all-MiniLM-L12-v2`, `all-mpnet-base-v2` and `LaBSE`, whose technical details are reported in Table 1.

| Model | Base Model | Number of Parameters | Reference |
| ----------------- | ------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| all-MiniLM-L12-v2 | MiniLM-L12-H384-uncased by Microsoft | 1B | https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2; **Paper**: Wang et al., 2020 |
| all-mpnet-base-v2 | mpnet-base by Microsoft | 1B | **HuggingFace**: https://huggingface.co/sentence-transformers/all-mpnet-base-v2; **Paper**: Song et al., 2020 |
| LaBSE | LaBSE by Google | 17B English sentence pairs + 6B Multi-Lingual Sentence Pairs | **HuggingFace**: https://huggingface.co/sentence-transformers/LaBSE, **Paper**: Feng et al., 2020 |

*Table 1*: Technical details and references for the evaluated embedding models

### Data and preparation

The evaluation data employed were the PDF version of two landmark papers in the field of AI: _Attention is all you need_ (Vaswani et al., 2017) and _Generative Adversarial Networks_ (Goodfellow et al., 2014).

The PDFs text was extracted via a PyPDF (https://pypdf.readthedocs.io/en/stable/; last visited 12-04-2024) wrapper offered by LangChain (https://langchain.com; last visited 12-04-2024). 

The extracted text was chunked in sizes of 500, 1000 and 1500 characters by LangChain text splitter.

These chunks were uploaded to a local Qdrant-operated vector database via `qdrant-client`, Qdrant's python bindings (https://pypi.org/project/qdrant-client/; last visited 12-04-2024).

The PDF preprocessing and uploading is integrated in SenTrEv framework.

### Evaluation workflow 

SenTrEv applies a very simple evaluation workflow:

1. After the PDF text extraction and chunking (cfr. _supra_) phase, the chunks are reduced according to a (optionally) user-defined percentage (default is 25%), which is randomly extracted at any point of each chunk.
2. The reduced chunks are mapped to their original ones in a dictionary
3. Each model encodes the original chunks and uploads the vectors to the Qdrant vector storage
4. The reduced chunks are then used as queries for retrieval
5. Starting from retrieval results, accuracy, time and carbon emissions statistics are calculated and plotted.

See Fig.1 for a visualization of the workflow

![workflow](https://raw.githubusercontent.com/AstraBert/SenTrEv-case-study/main/imgs/SenTrEv_Eval_Workflow.png)

_Fig. 1_: Evaluation workflow for SenTrEv

In this study, the text percentages for chunk reduction were set to 40, 60 and 80%.

Additionally, the distance metrics were tweaked, exploring all available ones for Qdrant, i.e. cosine, euclidean, Manhattan and dot product distance.

Accounting also for the different chunking sizes (cfr. _supra_), this study presents 36 total test runs, each with a combination of the aforementioned parameters. We then averaged across all these tests too see which model performed better and what factors mostly influenced their performance.

All the tests were conducted with SenTrEv v0.1.0 on Python 3.11.9, exploiting a Windows v10.0.22631.4460 machine.
Models were loaded on GPU for faster inference (GPU hardware: NVDIA RTX4050, 6GB ggr6-sdRAM; CUDA v12.3).

## Results

The metrics used to evaluate performance were:

- **Success rate**: defined as the number retrieval operation in which the correct context was retrieved ranking top among all the retrieved contexts, out of the total retrieval operations:

  $SR = \frac{Ncorrect}{Ntot}$ (eq.1)

- **Mean Reciprocal Ranking (MRR)**: MRR defines how high in ranking the correct context is placed among the retrieved results. MRR@10 was used, meaning that for each retrieval operation 10 items were returned and an evaluation was carried out for the ranking of the correct context, which was then normalized between 0 and 1 (already implemented in SenTrEv). An MRR of 1 means that the correct context was ranked first, whereas an MRR of 0 means that it wasn't retrieved. MRR is calculated with the following general equation:

  $MRR = \frac{ranking + Nretrieved - 1}{Nretrieved}$ (eq.2)

  When the correct context is not retrieved, MRR is automatically set to 0. MRR is calculated for each retrieval operation, then the average and standard deviation are calculated and reported.
- **Time performance**: for each retrieval operation the time performance in seconds is calculated: the average and standard deviation are then reported.
- **Carbon emissions**: Carbon emissions are calculated in gCO2eq (grams of CO2 equivalent) through the Python library `codecarbon` (https://codecarbon.io/; last visited 12-04-2024) and were evaluated for the Austrian region. They are reported for the global computational load of all the retrieval operations.



### Embedders performance evaluation


### Factors influencing perfomance



![Figure 2](https://raw.githubusercontent.com/AstraBert/SenTrEv-case-study/main/imgs/Figure_2.png)
_Fig. 2_: Factors influencing model performances: **A** is chunking size, **B** is text percentage and **C** is distance metrics

### Code and data availability statement

All code and data are contained in the dedicated GitHub repository: https://github.com/AstraBert/SenTrEv-case-study

## Conclusion

[^1]: Deparment of Biology and Biotechnology, University of Pavia
