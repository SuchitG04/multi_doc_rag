# Multi-Doc RAG

## About

Query multiple documements using RAG! RAG is usually performed on a single document source or the entire knowledge base is treated as a single source. This RAG application can do much more than that!

The primary usecase this was built for was to query and compare 10K reports. An example query might look like this: "How much are Apple and AMD investing in R&D?". Now this question cannot be treated as a single question because information from both the AMD and Apple 10K report is required. This application accommodates for this.

## Tools used:

- LangChain
- ElasticSearch VectorDB
- GPT3.5

## Usage

Check out [this](blog_nb.ipynb) notebook to get started. You'll find the end-to-end implementation of RAG here.

[This](experiments) folder has the spaghetti code written by me during experimentation and breaking my head while implementing this.

## Explanation

Check out the accompanying blog post over at []().

Note: The packages installed in the blog post should suffice to run the [notebook](blog_nb.ipynb). I strongly reccomend you to not use my [requirements.txt](requirements.txt) because it has a lot of uneccessary packages and you will end up downloading a lot more packages than necessary.
