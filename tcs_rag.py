import time
import numpy as np


def basic_rag(question, chroma_collection, client):
    """
    Basic RAG: retrieve 5 chunks, generate answer with GPT-4.1
    Returns: {"answer": str, "runtime": float}
    """
    start_time = time.time()

    # Retrieve 5 most relevant chunks
    results = chroma_collection.query(query_texts=[question], n_results=5)

    if not results['documents'][0]:
        runtime = time.time() - start_time
        return {
            "answer": "No relevant information found in TCS report.",
            "runtime": round(runtime, 2)
        }

    # Combine chunks into context
    context = "\n\n".join(results['documents'][0])

    # Generate answer with GPT-4.1
    try:
        response = client.responses.create(
            model="gpt-4.1",
            input=f"""Based on the following excerpts from the TCS Annual Report, please answer this question: {question}

Context from TCS Annual Report:
{context}

Please provide a clear, accurate answer based only on the information provided above. If the context doesn't contain enough information to fully answer the question, please say so."""
        )

        answer = response.output_text if hasattr(response, 'output_text') else str(response)
        runtime = time.time() - start_time

        return {
            "answer": answer.strip(),
            "runtime": round(runtime, 2)
        }

    except Exception as e:
        runtime = time.time() - start_time
        return {
            "answer": f"Error calling OpenAI API: {str(e)}",
            "runtime": round(runtime, 2)
        }


def query_expansion_rag(question, chroma_collection, client):
    """
    Query Expansion RAG: generate hypothetical answer, combine with question, retrieve 5 chunks
    Returns: {"answer": str, "runtime": float, "hypothetical_answer": str}
    """
    start_time = time.time()

    # Step 1: Generate hypothetical answer
    try:
        hyp_response = client.responses.create(
            model="gpt-4.1",
            input=f"""You are a helpful expert financial research assistant. Provide an example answer to the given question, that might be found in a document like an annual report.

Question: {question}

Generate a realistic, detailed answer that would typically appear in an annual report:"""
        )

        hypothetical_answer = hyp_response.output_text if hasattr(hyp_response, 'output_text') else str(hyp_response)
        hypothetical_answer = hypothetical_answer.strip()

    except Exception as e:
        # Fallback to original question if hypothetical generation fails
        hypothetical_answer = ""

    # Step 2: Create expanded query (original + hypothetical)
    if hypothetical_answer:
        expanded_query = f"{question} {hypothetical_answer}"
    else:
        expanded_query = question

    # Step 3: Retrieve 5 chunks using expanded query
    results = chroma_collection.query(query_texts=[expanded_query], n_results=5)

    if not results['documents'][0]:
        runtime = time.time() - start_time
        return {
            "answer": "No relevant information found in TCS report.",
            "runtime": round(runtime, 2),
            "hypothetical_answer": hypothetical_answer
        }

    # Step 4: Generate final answer using ONLY retrieved context (not hypothetical)
    context = "\n\n".join(results['documents'][0])

    try:
        response = client.responses.create(
            model="gpt-4.1",
            input=f"""Based on the following excerpts from the TCS Annual Report, please answer this question: {question}

Context from TCS Annual Report:
{context}

Please provide a clear, accurate answer based only on the information provided above. If the context doesn't contain enough information to fully answer the question, please say so."""
        )

        answer = response.output_text if hasattr(response, 'output_text') else str(response)
        runtime = time.time() - start_time

        return {
            "answer": answer.strip(),
            "runtime": round(runtime, 2),
            "hypothetical_answer": hypothetical_answer
        }

    except Exception as e:
        runtime = time.time() - start_time
        return {
            "answer": f"Error calling OpenAI API: {str(e)}",
            "runtime": round(runtime, 2),
            "hypothetical_answer": hypothetical_answer
        }


def multiple_queries_rag(question, chroma_collection, client, cross_encoder):
    """
    Multiple Queries RAG: generate 5 related questions, retrieve chunks, cross-encoder re-rank, take top 5
    Returns: {"answer": str, "runtime": float, "generated_queries": list}
    """
    start_time = time.time()

    # Step 1: Generate 5 related questions
    try:
        queries_response = client.responses.create(
            model="gpt-4.1",
            input=f"""You are a helpful expert financial research assistant. Your users are asking questions about the TCS Annual Report.

Suggest up to 5 additional related questions to help them find the information they need, for the provided question.
Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic.
Make sure they are complete questions, and that they are related to the original question.
Output one question per line. Do not number the questions.

Original question: {question}

Generate 5 related questions:"""
        )

        content = queries_response.output_text if hasattr(queries_response, 'output_text') else str(queries_response)
        generated_queries = [q.strip() for q in content.split("\n") if q.strip()]

        # Ensure we have exactly 5 questions
        if len(generated_queries) > 5:
            generated_queries = generated_queries[:5]
        elif len(generated_queries) < 5:
            while len(generated_queries) < 5:
                generated_queries.append(generated_queries[0] if generated_queries else question)

    except Exception as e:
        generated_queries = []

    # Step 2: Create list of all queries (original + 5 related)
    all_queries = [question] + generated_queries

    # Step 3: Retrieve chunks from all queries and deduplicate
    all_chunks = []
    for query in all_queries:
        try:
            results = chroma_collection.query(query_texts=[query], n_results=10)  # Get more for better cross-encoder selection
            if results['documents'][0]:
                all_chunks.extend(results['documents'][0])
        except Exception as e:
            continue

    # Step 4: Deduplicate chunks using exact string matching
    unique_chunks = []
    seen_chunks = set()
    for chunk in all_chunks:
        if chunk not in seen_chunks:
            unique_chunks.append(chunk)
            seen_chunks.add(chunk)

    if not unique_chunks:
        runtime = time.time() - start_time
        return {
            "answer": "No relevant information found in TCS report.",
            "runtime": round(runtime, 2),
            "generated_queries": generated_queries
        }

    # Step 5: Cross-encoder re-ranking - score all chunks against original question
    pairs = [[question, chunk] for chunk in unique_chunks]
    scores = cross_encoder.predict(pairs)

    # Step 6: Get top 5 highest scoring chunks
    top_indices = np.argsort(scores)[::-1][:5]  # Top 5 indices
    top_chunks = [unique_chunks[i] for i in top_indices]

    # Step 7: Generate final answer using top 5 re-ranked chunks
    context = "\n\n".join(top_chunks)

    try:
        response = client.responses.create(
            model="gpt-4.1",
            input=f"""Based on the following excerpts from the TCS Annual Report, please answer this question: {question}

Context from TCS Annual Report:
{context}

Please provide a clear, accurate answer based only on the information provided above. If the context doesn't contain enough information to fully answer the question, please say so."""
        )

        answer = response.output_text if hasattr(response, 'output_text') else str(response)
        runtime = time.time() - start_time

        return {
            "answer": answer.strip(),
            "runtime": round(runtime, 2),
            "generated_queries": generated_queries
        }

    except Exception as e:
        runtime = time.time() - start_time
        return {
            "answer": f"Error calling OpenAI API: {str(e)}",
            "runtime": round(runtime, 2),
            "generated_queries": generated_queries
        }