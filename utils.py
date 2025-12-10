from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
from PIL import Image

def create_pdf(report_text, filename, modality, body_part, image=None, language="English"):
    """Generate a PDF report with the medical image and report text.
    
    Args:
        report_text: The medical report text
        filename: Original image filename
        modality: Detected imaging modality
        body_part: Detected body part
        image: PIL Image object (optional)
        language: Report language (currently only affects metadata)
    
    Returns:
        PDF file as bytes
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "Medical Imaging Report")

    # Metadata
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"File Name: {filename}")
    c.drawString(50, height - 100, f"Modality: {modality}")
    c.drawString(50, height - 120, f"Body Part: {body_part}")

    # Add image if provided
    y_position = height - 140
    if image is not None:
        try:
            # Convert PIL Image to ImageReader
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            img_reader = ImageReader(img_buffer)
            
            # Calculate image dimensions (max width 250, maintain aspect ratio)
            img_width, img_height = image.size
            max_width = 250
            aspect_ratio = img_height / img_width
            display_width = min(max_width, img_width)
            display_height = display_width * aspect_ratio
            
            # Draw image
            c.drawImage(img_reader, 50, y_position - display_height - 10, 
                       width=display_width, height=display_height, 
                       preserveAspectRatio=True)
            
            y_position = y_position - display_height - 30
        except Exception as e:
            # If image fails, just continue without it
            c.setFont("Helvetica", 10)
            c.drawString(50, y_position - 20, f"[Image could not be embedded: {str(e)}]")
            y_position = y_position - 40

    # Report Content
    c.line(50, y_position, width - 50, y_position)
    y_position -= 20
    
    text_object = c.beginText(50, y_position)
    text_object.setFont("Helvetica", 11)
    
    # Handle multiline text
    lines = report_text.split('\n')
    for line in lines:
        # Check if we need a new page
        if text_object.getY() < 50:
            c.drawText(text_object)
            c.showPage()
            text_object = c.beginText(50, height - 50)
            text_object.setFont("Helvetica", 11)
        
        # Simple wrapping for long lines
        if len(line) > 80:
            words = line.split()
            current_line = ""
            for word in words:
                if len(current_line + word) < 80:
                    current_line += word + " "
                else:
                    text_object.textLine(current_line.strip())
                    current_line = word + " "
            if current_line:
                text_object.textLine(current_line.strip())
        else:
            text_object.textLine(line)
        
    c.drawText(text_object)
    
    c.showPage()
    c.save()
    
    buffer.seek(0)
    return buffer.getvalue()

def translate_to_german(text: str) -> str:
    """Translate English text to German using the local LM with fallback dictionary.
    
    Args:
        text: English text to translate
        
    Returns:
        German translated text
    """
    from agents.agent import configure_dspy
    import dspy
    configure_dspy()
    prompt = f"Translate the following medical report from English to German:\n\n{text}\n\nGerman translation:"
    
    # Fallback dictionary
    translations = {
        "FINDINGS": "BEFUNDE",
        "IMPRESSION": "EINDRUCK",
        "RECOMMENDATIONS": "EMPFEHLUNGEN",
        "examination": "Untersuchung",
        "shows": "zeigt",
        "normal": "normal",
        "anatomical": "anatomische",
        "structures": "Strukturen",
        "No acute abnormalities detected": "Keine akuten Anomalien festgestellt",
        "study": "Studie",
        "No significant pathological findings identified": "Keine signifikanten pathologischen Befunde identifiziert",
        "Continue routine clinical follow-up as needed": "Routinemäßige klinische Nachsorge nach Bedarf fortsetzen",
        "Correlate with clinical symptoms": "Mit klinischen Symptomen korrelieren",
        "No immediate intervention required": "Keine sofortige Intervention erforderlich",
        "The": "Die",
        "of the": "der",
    }

    try:
        response = dspy.settings.lm(prompt)
        translated_text = response[0]
        
        # Validation: Check if key German terms are present
        if "BEFUNDE" in translated_text or "EINDRUCK" in translated_text:
            return translated_text
        else:
            print("Model output did not contain expected German keywords. Using fallback.")
            raise ValueError("Model failed to translate keywords")

    except Exception:
        german_text = text
        for eng, ger in translations.items():
            german_text = german_text.replace(eng, ger)
        return german_text
