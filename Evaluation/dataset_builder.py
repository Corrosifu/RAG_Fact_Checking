from ragas import EvaluationDataset

class DatasetBuilder:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def build(self, test_set: list) -> EvaluationDataset:
        records = []
        for item in test_set:
            query = item["query"]
            expected = item["expected"]
            docs = self.retriever.retrieve(query)
            contexts = [d.page_content for d in docs]
            answer = self.generator.generate(query, docs)

            records.append({
                "user_input": query,        
                "retrieved_contexts": contexts,
                "response": answer,
                "reference": expected,
            })

        return EvaluationDataset.from_list(records)
