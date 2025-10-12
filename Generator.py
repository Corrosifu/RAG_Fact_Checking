import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optimize CUDA memory allocation for large model inference.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,expandable_segments:True"


class Generator:
    """
    Pedagogical scientific explanation generator.

    This class wraps a lightweight LLaMA-style model (or any HuggingFace causal model)
    to produce clear, educational explanations of scientific or machine learning topics.
    It automatically formats the prompt, cleans retrieved context text,
    and handles device placement for efficient inference.

    Attributes
    ----------
    device : str
        The device used for inference ('cpu' or 'cuda').
    max_context_chars : int
        Maximum number of characters from the context to include in the prompt.
    tokenizer : AutoTokenizer
        HuggingFace tokenizer corresponding to the selected model.
    model : AutoModelForCausalLM
        The loaded causal language model for text generation.

    Example
    -------
    >>> generator = Generator()
    >>> answer = generator.generate("What is gradient descent?", context=[])
    >>> print(answer)
    'Gradient descent is an optimization algorithm...'
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        device: str = None,
        max_context_chars: int = 1500
    ):
        """
        Initialize the generator model and tokenizer.

        Parameters
        ----------
        model_name : str, optional
            HuggingFace model name or path. Defaults to a lightweight LLaMA instruction-tuned model.
        device : str, optional
            Computation device ('cuda' or 'cpu'). Automatically detects GPU if available.
        max_context_chars : int, optional
            Maximum number of context characters to include in the prompt. Default is 1500.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_context_chars = max_context_chars

        # --- Model loading ---
        print(f"📦 Loading model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        print("✅ Model loaded successfully.")

    # ----------------------------
    # Text preprocessing
    # ----------------------------
    def _clean_context(self, text: str) -> str:
        """
        Clean and normalize context text to avoid irrelevant noise in prompts.

        Removes excessive whitespace, citation-like parentheses, and
        non-alphanumeric characters (except basic punctuation).

        Parameters
        ----------
        text : str
            Raw context text.

        Returns
        -------
        str
            Cleaned and normalized context string.
        """
        # Remove redundant spaces and citation patterns like "(Smith, 2021)"
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\(.*?\d{4}.*?\)', '', text)
        # Keep only clean alphanumeric characters and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9.,;:\-\s]', '', text)
        return text.strip()

    # ----------------------------
    # Prompt formatting
    # ----------------------------
    def format_prompt(self, query: str, context: list) -> str:
        """
        Format the instructional prompt for scientific explanation generation.

        The prompt guides the model to:
          - Teach complex concepts simply
          - Use examples
          - Avoid copying from the retrieved context

        Parameters
        ----------
        query : str
            User's question or topic.
        context : list
            List of `Document` objects (from LangChain or similar), providing background text.

        Returns
        -------
        str
            A structured prompt ready for model input.
        """
        # Clean each context chunk and join into one text block
        context_text = "\n".join([self._clean_context(item.page_content) for item in context])
        # Truncate context to limit token size
        context_text = context_text[:self.max_context_chars]

        # Construct a pedagogical prompt
        return (
            "You are a friendly and knowledgeable scientific teaching assistant. "
            "Explain complex machine learning or scientific concepts clearly and concisely, "
            "using simple language and examples when appropriate. Do not copy text from the context; "
            "instead, explain it in your own words.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
        )

    # ----------------------------
    # Generation
    # ----------------------------
    def generate(self, query: str, context: list, max_tokens: int = 224) -> str:
        """
        Generate a pedagogical answer to a scientific or ML question.

        Performs full prompt formatting, tokenization, model inference, and decoding.

        Parameters
        ----------
        query : str
            The scientific or technical question to answer.
        context : list
            List of `Document` objects providing related information.
        max_tokens : int, optional
            Maximum number of tokens to generate. Default is 224.

        Returns
        -------
        str
            The model-generated explanatory answer.
        """
        # Prepare full text prompt with context
        prompt = self.format_prompt(query, context)
        # Tokenize and move tensors to target device
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)

        # Generate output without gradient computation (for efficiency)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Extract only the newly generated tokens (excluding the prompt itself)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = output[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()
