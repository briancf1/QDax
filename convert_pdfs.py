from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

converter = PdfConverter(
    artifact_dict=create_model_dict(),
)
FILEPATH = "/Users/briancf/Desktop/source/EvoAlgsAndSwarm/lib-qdax/QDax/research_papers/23-1027.pdf"
rendered = converter(FILEPATH)
text, _, images = text_from_rendered(rendered)

pdf_name = FILEPATH.split("/")[-1].replace(".pdf", "")
with open(f"{pdf_name}.md", "w") as f:
    f.write(text)