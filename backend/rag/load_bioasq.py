from pathlib import Path
from app.rag.xml_parser import parse_bioasq_xml

def load_all_bioasq(base_path="app/data/bioasq"):
    documents = []

    for xml_file in Path(base_path).rglob("*.xml"):
        qa_pairs = parse_bioasq_xml(str(xml_file))
        for qa in qa_pairs:
            qa["source"] = f"BioASQ | {xml_file.parent.name}"
            documents.append(qa)

    print(f"BioASQ loaded: {len(documents)} QA pairs")
    return documents
