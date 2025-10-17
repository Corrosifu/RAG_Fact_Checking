import datetime
from langsmith import Client
import pandas as pd
from config import DEFAULT_CSV_PATH
class EvalLogger:
    def __init__(self):
        self.client = Client()

    def log_to_langsmith(self, df: pd.DataFrame, experiment_name="RAG Evaluation"):
        run_id = f"{experiment_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"

        self.client.create_run(
            name=run_id,
            run_type="chain",  
            inputs={},
            outputs={"results": df.to_dict(orient="records")},
            metadata={"rows": len(df)}
            )



    def save_csv(self, df: pd.DataFrame, path=DEFAULT_CSV_PATH):
        df.to_csv(path, index=False)
        print(f"✅ Results saved to {path}")
