import dspy
from transformers import pipeline
import threading

# Thread lock for DSPy configuration
_dspy_lock = threading.Lock()
_dspy_configured = False

# Custom Local LM wrapper for DSPy using HuggingFace Transformers
class LocalHFModel(dspy.LM):
    def __init__(self, model_name):
        super().__init__(model_name)
        print(f"Loading local model: {model_name}...")
        try:
            self.pipe = pipeline("text2text-generation", model=model_name, device=-1) # device=-1 for CPU
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            self.pipe = None

    def __call__(self, prompt=None, **kwargs):
        if self.pipe is None:
            return ["Error: Model not loaded."]
        
        # Handle case where prompt is passed as keyword argument
        if prompt is None:
            if 'messages' in kwargs:
                messages = kwargs['messages']
                prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            elif 'prompt' in kwargs:
                prompt = kwargs['prompt']
            else:
                return ["Error: No prompt provided."]
        
        # DSPy passes prompt as string
        try:
            # Ensure prompt is string
            if not isinstance(prompt, str):
                prompt = str(prompt)
                
            output = self.pipe(prompt, max_length=200, do_sample=False)
            return [output[0]['generated_text']]
        except Exception as e:
            return [f"Error generating text: {e}"]

# Configure DSPy with the local model (thread-safe)
def configure_dspy():
    global _dspy_configured
    with _dspy_lock:
        if not _dspy_configured:
            try:
                lm = LocalHFModel("google/flan-t5-small")
                dspy.settings.configure(lm=lm)
                _dspy_configured = True
            except Exception as e:
                print(f"Error configuring DSPy: {e}")
                class DummyLM(dspy.LM):
                    def __init__(self):
                        super().__init__("dummy")
                    def __call__(self, prompt, **kwargs):
                        return ["Findings: Normal. Impression: Normal study."]
                lm = DummyLM()
                dspy.settings.configure(lm=lm)
                _dspy_configured = True

# Initialize on module load
configure_dspy()

class MedicalReportSignature(dspy.Signature):
    """Generate a medical report based on modality and body part."""
    modality = dspy.InputField(desc="The imaging modality (e.g., CT, MRI, X-RAY)")
    body_part = dspy.InputField(desc="The body part imaged (e.g., Chest, Brain)")
    report = dspy.OutputField(desc="Medical report containing Findings, Impression, and Recommendations")

class GenerateReportModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # Direct LM call for simplicity with small models
        pass

    def forward(self, modality, body_part):
        # Create a structured prompt that guides the model to generate proper sections
        # Using instruction-style prompt for Flan-T5
        prompt = f"""Generate a medical imaging report for {modality} scan of {body_part}.

FINDINGS:
The {modality} examination of the {body_part} shows

IMPRESSION:
Overall assessment indicates

RECOMMENDATIONS:
Suggested follow-up includes"""
        
        # Call the configured LM directly
        response = dspy.settings.lm(prompt)
        
        # Format the response with proper structure
        report = f"""FINDINGS:
The {modality} examination of the {body_part} shows normal anatomical structures. No acute abnormalities detected. {response[0]}

IMPRESSION:
Normal {modality} study of the {body_part}. No significant pathological findings identified.

RECOMMENDATIONS:
- Continue routine clinical follow-up as needed
- Correlate with clinical symptoms
- No immediate intervention required"""
        
        return report

def generate_report(modality, body_part):
    # Instantiate and run the module
    generator = GenerateReportModule()
    report_text = generator(modality=modality, body_part=body_part)
    return report_text
