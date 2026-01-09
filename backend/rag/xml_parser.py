import xml.etree.ElementTree as ET

def parse_bioasq_xml(xml_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    records = []

    for qa in root.iter("QAPair"):
        question = qa.findtext("Question")
        answer = qa.findtext("Answer")

        if not question or not answer:
            continue

        records.append({
            "text": f"Q: {question}\nA: {answer}",
            "source": "BioASQ"
        })

    return records
