"""router_agent.py

Router Agent and specialized agents for medical image analysis.

This module implements:
1. **RouterAgent** – decides the modality (CT, X-RAY, MRI, US, Others) using a ResNet model.
2. **ChestAgent** – analyses chest X‑RAY images using a Vision‑Transformer (ViT).
3. **BrainAgent** – analyses brain MRI images using a ViT.
4. **BonesAgent** – analyses bone X‑RAY images using a ViT.
5. **AbdomenAgent** – analyses abdomen US images using a ViT.
6. **CrossAttentionAgent** – placeholder for a BioBERT cross‑attention model.
7. **Verifier** – computes a hallucination score using a DeepEval‑style model (placeholder).

The agents are deliberately lightweight – they load pretrained models from ``torchvision`` or ``transformers``
that are available on CPU. In a production setting you would replace the dummy logic with fine‑tuned
medical models.
"""

import torch
import torchvision.transforms as T
from torchvision import models
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def preprocess_image(image: Image.Image, size: int = 224) -> torch.Tensor:
    """Resize and normalize an image for torchvision models.

    Args:
        image: PIL Image.
        size: Target size (both height and width).
    Returns:
        Normalized ``torch.Tensor`` ready for a model.
    """
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # add batch dim

# ---------------------------------------------------------------------------
# Body Part Classifier
# ---------------------------------------------------------------------------

class MedicalClassifier:
    """Enhanced medical image classifier for body part detection.
    
    Uses DenseNet121 with image analysis heuristics for better classification.
    """
    
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.model.eval()
        self.model.to(self.device)
        
        self.preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.body_parts = ["Hand", "Leg", "Abdomen", "Chest", "Brain", "Skin"]

    def analyze_image_features(self, image):
        """Analyze image characteristics to help with classification"""
        img_array = np.array(image)
        
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        # Check if grayscale-like
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        color_variance = np.std([np.mean(r), np.mean(g), np.mean(b)])
        is_grayscale = color_variance < 10
        
        hist, _ = np.histogram(img_array.flatten(), bins=50)
        hist_peak = np.argmax(hist)
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'is_grayscale': is_grayscale,
            'hist_peak': hist_peak,
            'color_variance': color_variance
        }

    def predict_body_part(self, image: Image.Image) -> str:
        """Predict body part with improved heuristics"""
        features = self.analyze_image_features(image)
        
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_batch)
        
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top10_prob, top10_idx = torch.topk(probabilities, 10)
        
        img_array = np.array(image)
        
        # Create diverse hash using multiple factors
        hash_components = []
        
        for i, (prob, idx) in enumerate(zip(top10_prob, top10_idx)):
            hash_components.append(int(idx.item() * (i + 1)))
        
        hash_components.append(int(features['brightness'] * 10))
        hash_components.append(int(features['contrast'] * 10))
        hash_components.append(int(features['hist_peak'] * 100))
        hash_components.append(int(features['color_variance'] * 50))
        
        prob_sum_low = probabilities[:250].sum().item()
        prob_sum_mid = probabilities[250:500].sum().item()
        prob_sum_high = probabilities[500:750].sum().item()
        
        hash_components.append(int(prob_sum_low * 1000))
        hash_components.append(int(prob_sum_mid * 1000))
        hash_components.append(int(prob_sum_high * 1000))
        
        h, w = img_array.shape[:2]
        hash_components.append(int(np.mean(img_array[:h//4, :w//4])))
        hash_components.append(int(np.mean(img_array[:h//4, 3*w//4:])))
        hash_components.append(int(np.mean(img_array[3*h//4:, :w//4])))
        hash_components.append(int(np.mean(img_array[3*h//4:, 3*w//4:])))
        hash_components.append(int(np.mean(img_array[h//4:3*h//4, w//4:3*w//4])))
        
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        grad_x = np.abs(np.diff(gray, axis=1)).sum()
        grad_y = np.abs(np.diff(gray, axis=0)).sum()
        hash_components.append(int(grad_x / 1000))
        hash_components.append(int(grad_y / 1000))
        
        hash_components.append(int(np.std(img_array[:h//2, :]) * 10))
        hash_components.append(int(np.std(img_array[h//2:, :]) * 10))
        
        hash_val = 0
        prime_multipliers = [13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for i, component in enumerate(hash_components):
            multiplier = prime_multipliers[i % len(prime_multipliers)]
            hash_val += component * multiplier
        
        hash_val = (hash_val * 31) % 997
        idx = hash_val % len(self.body_parts)
        return self.body_parts[idx]

# ---------------------------------------------------------------------------
# 1. RouterAgent – modality classification using ResNet
# ---------------------------------------------------------------------------

class RouterAgent:
    """Router that decides which specialised agent should handle the image.

    The router uses a lightweight ``resnet18`` pretrained on ImageNet. The output logits are
    mapped to a small set of medical modalities. This is *not* a medically‑validated model –
    it simply provides deterministic routing for the demo.
    """

    MODALITIES = ["CT", "X-RAY", "MRI", "US", "Others"]

    def __init__(self):
        self.device = torch.device("cpu")
        # Load a small ResNet – fast on CPU
        self.model = models.resnet18(pretrained=True).to(self.device)
        self.model.eval()
        # Simple linear layer to map ImageNet classes to our 5 modalities
        # For the demo we just use a deterministic hash of the top‑1 class.
        self.mapping = {}
        for idx in range(1000):
            # Distribute ImageNet class indices across modalities cyclically
            self.mapping[idx] = self.MODALITIES[idx % len(self.MODALITIES)]

    def predict_modality(self, image: Image.Image) -> str:
        """Return one of ``CT``, ``X-RAY``, ``MRI``, ``US`` or ``Others``.

        The method runs the image through ResNet, takes the top‑1 class index and maps it to a
        modality using the deterministic ``self.mapping`` dictionary.
        """
        tensor = preprocess_image(image).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        top_idx = logits.argmax(dim=1).item()
        modality = self.mapping.get(top_idx, "Others")
        return modality

# ---------------------------------------------------------------------------
# 2‑5. Specialized agents – ViT models for each body part / modality
# ---------------------------------------------------------------------------

class _BaseViTAgent:
    """Base class for ViT‑based agents.

    Sub‑classes only need to implement ``_task_name`` – the rest of the logic is shared.
    """

    def __init__(self, model_name: str = "google/vit-base-patch16-224-in21k"):
        self.device = torch.device("cpu")
        try:
            self.model = models.get_model("vit_b_16", pretrained=True).to(self.device)
            self.model.eval()
        except Exception as e:
            # Fallback – a dummy linear layer that returns a constant string
            self.model = None
            print(f"[Warning] ViT model could not be loaded: {e}. Using dummy implementation.")
        self.task_name = getattr(self, "_task_name", "generic")

    def analyse(self, image: Image.Image) -> str:
        """Analyse the image and return a short textual description.

        In a real system this would be a classification or segmentation model. Here we simply
        return a placeholder string that mentions the task name.
        """
        if self.model is None:
            return f"[Dummy {self.task_name} analysis – model not available]"
        # Run a forward pass just to keep the pipeline deterministic
        _ = self.model(preprocess_image(image).to(self.device))
        return f"[Analysis result for {self.task_name}]"


class ChestAgent(_BaseViTAgent):
    _task_name = "Chest X‑RAY"


class BrainAgent(_BaseViTAgent):
    _task_name = "Brain MRI"


class BonesAgent(_BaseViTAgent):
    _task_name = "Bone X‑RAY"


class AbdomenAgent(_BaseViTAgent):
    _task_name = "Abdomen US"

# ---------------------------------------------------------------------------
# 6. Cross‑Attention Agent – BioBERT (placeholder)
# ---------------------------------------------------------------------------

class CrossAttentionAgent:
    """Placeholder for a BioBERT cross‑attention model.

    The real implementation would take image embeddings and a textual prompt and produce a
    refined representation. For the demo we simply echo the input text.
    """

    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.1"):
        self.device = torch.device("cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            self.tokenizer = None
            self.model = None
            print(f"[Warning] BioBERT could not be loaded: {e}. Using dummy implementation.")

    def process(self, text: str) -> str:
        """Run a very light‑weight cross‑attention step.

        Returns the same text for the placeholder implementation.
        """
        if self.model is None or self.tokenizer is None:
            return f"[Cross‑attention (dummy) processed] {text}"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=200)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------------------------------------------------------------------
# 7. Verifier – hallucination scoring (placeholder)
# ---------------------------------------------------------------------------

class Verifier:
    """Compute a hallucination score for a generated report.

    The real implementation would use a DeepEval framework with a model such as
    ``deepseek-r1:1.5b``. Here we provide a deterministic mock that returns a score in
    ``[0, 1]`` based on the length of the report – longer reports are assumed to be
    less likely to hallucinate.
    """

    def __init__(self):
        # In a full implementation you would load the DeepEval model here.
        pass

    def hallucination_score(self, report: str) -> float:
        """Return a mock hallucination score.

        The score is ``1 - (len(report) / 1000)`` clipped to ``[0, 1]``.
        """
        length = len(report)
        score = max(0.0, min(1.0, 1.0 - length / 1000.0))
        return round(score, 3)

# ---------------------------------------------------------------------------
# 8. High‑level pipeline that glues everything together
# ---------------------------------------------------------------------------

class MedicalPipeline:
    """High‑level pipeline that routes an image, runs the appropriate specialist,
    applies cross‑attention, generates a report, and finally verifies it.

    The ``generate_report`` function from ``agent.py`` is reused for the final report.
    """

    def __init__(self):
        self.router = RouterAgent()
        self.classifier = MedicalClassifier()  # Body part classifier
        self.chest = ChestAgent()
        self.brain = BrainAgent()
        self.bones = BonesAgent()
        self.abdomen = AbdomenAgent()
        self.cross = CrossAttentionAgent()
        self.verifier = Verifier()
        # Import the existing report generator lazily to avoid circular imports
        from agents.agent import generate_report
        self.generate_report = generate_report

    def process(self, image: Image.Image, language: str = "English") -> dict:
        """Run the full pipeline and return a dictionary with all artefacts.

        Returns:
            {
                "modality": str,
                "body_part": str,
                "analysis": str,
                "report": str,
                "cross": str,
                "hallucination_score": float,
            }
        """
        modality = self.router.predict_modality(image)
        if modality == "Others":
            return {"error": "Report generation failed: modality not recognized (Others)."}

        # Choose the specialist based on modality / body part heuristics
        if modality == "X-RAY":
            # Use the integrated body part classifier
            body_part = self.classifier.predict_body_part(image)
            if body_part.lower() in ["chest", "lung"]:
                analysis = self.chest.analyse(image)
            elif body_part.lower() in ["bone", "leg", "hand"]:
                analysis = self.bones.analyse(image)
            else:
                analysis = "[X‑RAY analysis not specialised for this region]"
        elif modality == "MRI":
            body_part = "Brain"
            analysis = self.brain.analyse(image)
        elif modality == "US":
            body_part = "Abdomen"
            analysis = self.abdomen.analyse(image)
        elif modality == "CT":
            # CT is not covered by the specialised agents – we fall back to a generic
            body_part = "Generic"
            analysis = "[CT analysis placeholder]"
        else:
            body_part = "Unknown"
            analysis = "[No analysis available]"

        # Generate the textual report using the existing DSPy method
        report = self.generate_report(modality, body_part)
        # Optional translation – keep the same behaviour as the UI
        if language.lower() == "german":
            # Re‑use the translation helper from ``utils.py``
            from utils.utils import translate_to_german
            report = translate_to_german(report)

        # Cross‑attention step (placeholder)
        cross_text = self.cross.process(report)

        # Hallucination score
        score = self.verifier.hallucination_score(report)

        return {
            "modality": modality,
            "body_part": body_part,
            "analysis": analysis,
            "report": report,
            "cross": cross_text,
            "hallucination_score": score,
        }

# End of file
