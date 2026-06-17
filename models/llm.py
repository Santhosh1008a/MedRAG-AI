import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline
from threading import Thread

class Phi3MiniLLM:
    """
    Wrapper for Microsoft Phi-3 Mini (3.8B parameters).
    
    Explanation:
    - Why Phi-3 Mini? Phi-3 Mini punches far above its weight class. It exhibits reasoning capabilities 
      comparable to much larger models (like Mixtral or early GPT-3.5) while fitting comfortably in VRAM.
      It was chosen over Gemma 2B (which has weaker reasoning), TinyLlama (which hallucinates heavily), 
      and FLAN-T5 (which is encoder-decoder and less suited for open-ended conversational generation).
    - Quantization: We use bitsandbytes 4-bit quantization (NF4). Quantization reduces the precision of 
      the model weights from 16-bit floats to 4-bit, drastically reducing the VRAM required to load the model.
      This allows a 3.8B model to easily fit on a 4GB/8GB RTX 3050 GPU without significantly degrading quality.
    """
    def __init__(self, model_id: str = "microsoft/Phi-3-mini-4k-instruct", device_map: str = "auto"):
        self.model_id = model_id
        
        # Configure 4-bit Quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        )

        print(f"Loading Tokenizer: {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        
        print(f"Loading Model: {self.model_id} (4-bit quantized)")
        # For systems without a GPU, bitsandbytes will fail. 
        # We fallback to standard precision if CUDA is not available.
        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
            )
        else:
            print("Warning: CUDA not detected. Loading model in standard precision on CPU. This will be slow!")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="cpu",
                trust_remote_code=True
            )

        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        self.hf_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.1,     # Low temperature for factual medical RAG to prevent hallucination
            repetition_penalty=1.1,
            return_full_text=False,
            streamer=self.streamer
        )
        
        self.llm = HuggingFacePipeline(pipeline=self.hf_pipeline)

    def get_llm(self) -> HuggingFacePipeline:
        """Returns the LangChain-compatible LLM."""
        return self.llm

    def generate_stream(self, prompt: str):
        """
        Generates text while streaming tokens to the streamer.
        To use this, iterate over self.streamer in the main thread while this runs in a background thread.
        """
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        
        generation_kwargs = dict(
            inputs,
            streamer=self.streamer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
        )
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        return self.streamer
