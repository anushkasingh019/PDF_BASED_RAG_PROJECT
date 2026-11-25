import contextlib
import io
import unittest

from langchain_core.embeddings import FakeEmbeddings
from langchain_core.runnables import RunnableLambda

from main import (
    PDF_PATH,
    create_vectorstore,
    load_documents,
    query_rag,
    split_documents,
)


class RagPipelineTests(unittest.TestCase):
    def test_supplied_pdf_loads_and_splits_into_source_chunks(self):
        documents = load_documents()
        chunks = split_documents(documents)

        self.assertEqual(len(documents), 20)
        self.assertTrue(any("What is Java?" in doc.page_content for doc in documents))
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(chunk.page_content.strip() for chunk in chunks))
        self.assertTrue(all("page" in chunk.metadata for chunk in chunks))
        self.assertEqual(PDF_PATH.name, "Unit_3_Java_Questions_and_Answers.pdf")

    def test_missing_pdf_reports_the_requested_path(self):
        missing_path = PDF_PATH.parent / "missing.pdf"

        with self.assertRaises(FileNotFoundError) as error:
            load_documents(missing_path)
        self.assertIn(str(missing_path), str(error.exception))

    def test_empty_question_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Question must not be empty"):
            query_rag("   ", None, None)

    def test_pdf_context_flows_through_retrieval_and_answer_chain(self):
        source = load_documents()
        chunks = split_documents(source)
        vectorstore = create_vectorstore(chunks, FakeEmbeddings(size=384))
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        observed_prompts = []

        def fake_llm(prompt_value):
            observed_prompts.append(prompt_value.to_string())
            return "Java is a programming language."

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            answer = query_rag(
                "What is Java?",
                retriever,
                RunnableLambda(fake_llm),
            )

        self.assertEqual(answer, "Java is a programming language.")
        self.assertEqual(len(observed_prompts), 1)
        self.assertIn("What is Java?", observed_prompts[0])
        self.assertIn("Context:", observed_prompts[0])
        self.assertIn("Retrieved 3 relevant context chunk(s)", output.getvalue())


if __name__ == "__main__":
    unittest.main()
