# RAGs

Retrieval augmented generation is an AI framework that combines “retrieval” of facts from an external knowledge base with generative AI.

![alt text](<images/RAG flowchart.png>)
### Resources

Paper: [2020] by Meta: [https://arxiv.org/abs/2005.11401v4](https://arxiv.org/abs/2005.11401v4)
- [https://research.ibm.com/blog/retrieval-augmented-generation-RAG](https://research.ibm.com/blog/retrieval-augmented-generation-RAG)
- [https://cloud.google.com/use-cases/retrieval-augmented-generation?hl=en](https://cloud.google.com/use-cases/retrieval-augmented-generation?hl=en)
- [Implement Retrieval-Augmented Generation (RAG) to Accelerate LLM Application Development](https://www.intel.com/content/www/us/en/goal/how-to-implement-rag.html?cid=sem&source=sa360&campid=2025_ao_cbu_us_gmocoma_gmocrbu_awa_text-link_generic_exact_cd_HQ-COMM-EnterpriseAI-EG-EntAI-OBS_FC25023_google_b2b_is_non-pbm_intel&ad_group=RAG-General_Exact&intel_term=retrieval+augmented+generation&sa360id=43700081469669295&gad_source=1&gclid=Cj0KCQjwkN--BhDkARIsAD_mnIr9jCA1DI9mkEC9AptaC2l3ZKqgN5vybv8hjQ0mJrgo5g7ajEQNCuEaAkjsEALw_wcB&gclsrc=aw.ds)
- [https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/)
- [https://aws.amazon.com/what-is/retrieval-augmented-generation/](https://aws.amazon.com/what-is/retrieval-augmented-generation/)
- [https://www.promptingguide.ai/techniques/rag](https://www.promptingguide.ai/techniques/rag)

### Notes

In a nutshell, RAG searches for and fetches information relevant to a user’s query. The information is combined with the user’s prompt and is passed to the LLM. The LLM uses this augmented prompt to generate a response.

LLMs can be inconsistent - they have no idea what they are saying - they know how the words relate statistically - but not what they mean. They can be improved with RAGs by grounding the model on external sources of knowledge to supplement the LLMs internal representation of information.

Foundation models are fine-tuned to adapt to a variety of tasks - can be even fine-tuned for domain-specific knowledge. But fine-tuning alone rarely gives the model the full breadth of knowledge it needs to answer highly specific questions. RAG give access to LLMs to a specialized body of knowledge to answer questions in a more accurate way.

It’s like an<mark style="background: #BBFABBA6;"> open-book exam.</mark> LLM can refer to facts from the knowledge base rather than from memory.

> Do smaller LLMs perform better with RAGs than larger LLMs? Is this leveling the playing field?

When organizations try to adapt LLMs to their use cases - to boost up employee productivity or reduce operational costs - they need to customize the off-the-shelf model to their organization’s data. However, they will need to fine-tune the model which can be costly.

![alt text](<images/NVIDIA RAG flowchart.png>)

**How does it work?**
- It has two phases - **retrieval** and **content generation**.
- In the retrieval phase, powerful algorithms search for and retrieve snippets of information relevant to the user’s prompt or question. The facts can come from indexed documents on the internet or a narrow set of sources. This information undergoes pre-processing (tokenization, stemming, removal of stop words).
- This pre-processed information is appended to the user’s prompt and passed to the language model. The integration with the prompt enhances the context for the LLM.
- In the generative phase, LLM uses it’s internal representation and the external knowledge base to synthesize an engaging accurate answer.

**Vector databases** can efficiently index, store and retrieve information.
- Modern search engines now use vector databases to efficiently retrive information.
- Vector databases store documents as embeddings in a high-dimensional space, which allows for fast and accurate retrieval based on semantic similarity.
- Multi-modal embeddings can be used for images, audio and video.

*To RAG or to fine-tune?* - both approaches start with a foundational LLM. RAG can be useful when you don't have the resources to fine-tune the foundational model for your data. Though, RAG pipeline may include many computationally intensive components. End users often expect low-latency.

**Why is it beneficial?**

- Fresh information
    - RAG can improve accuracy and relevance of the LLM response. LLM will have access to the most current, reliable dacts.
- It can help LLM access internal knowledge bases (emails, documents, etc.) or external knowledge bases (scholarly articles, special datasets, etc.)
- Model’s answers can be cross referenced with the original content to verify.
- Reduce hallucinations - by grounding the LLM to a set of external, verifiable facts - model has fewer opportunities to pull information from it’s parameters - so less risk that the model will leak sensitive data or “hallucinate” or give misleading information.
- It also reduces the need for us to re-train the model on new updated data continuously. Thus, reducing computational and financial costs.
- LLMs need to know when to stop and not answer. LLMs with enough fine-tuning can be trained to pause, but it may need to see thousands of examples of questions that can and can’t be answered. RAG can help ground the LLM.

**Applications:**
Customer support: finding exact details for a specific order
Healthcare: Pull relevant patient information
HR queries from employee database - answering questions like can I take 3 more vacations in half day increments?

If employee wanted to know if she can take maternity leave - LLM which does not use RAG will cheerfully reply - suree - but instead the one with RAG will pause. Since the laws surrounding this issue vary from state to state and are complex.

**How to implement?**
https://ai.meta.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/

https://huggingface.co/facebook/rag-token-nq

**What is semantic search?**
Semantic search is a data searching technique that focuses on understanding the contextual meaning and intent behind user's search query - rather than only matching keywords. 

How does it work?
- Transforms both the query and documents into vector embeddings (usually using models like SBERT or CLIP).
- Measures the similarity between the query and documents using cosine similarity or other distance metrics.
- Returns the most relevant results from the database.

How is it different from RAG? RAG uses semantic search to find the relevant documents. 

| Aspect                | Semantic Search                     | RAG                                           |
| --------------------- | ----------------------------------- | --------------------------------------------- |
| **Purpose**           | Retrieve relevant docs              | Retrieve + generate response                  |
| **Output**            | List of relevant documents          | Natural language response                     |
| **Model Involvement** | Embedding model only                | Embedding + LLM for generation                |
| **Use Case**          | Search, recommendation              | QA systems, AI assistants                     |
| **Example**           | Search for “climate change impacts” | Ask “What are the impacts of climate change?” |

**What are knowledge graphs?**
Knowledge graph is a structured representation of real-world entities, their attributes, and the relationships between them. It organizes information in a graph format where - 
- **Nodes/Entities:** Represent objects, concepts, or entities (e.g., people, places, products).
- **Edges/Relationships:** Define the connections between entities (e.g., “works at,” “is a type of,” “located in”).
- **Attributes/Properties:** Describe additional information about the entities (e.g., name, age, date).

**What is CAG?** - Context-Augmented Generation
It's a variation of RAG that enhances the model's responses by incorporating external context <mark style="background: #ABF7F7A6;">beyond</mark> just retrieved documents. Uses broader context - text + structured data. RAG generally uses just text-based documents. 