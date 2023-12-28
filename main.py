import PyPDF2
from transformers import pipeline

model_name = "consciousAI/question-answering-roberta-base-s-v2"
genie = pipeline('question-answering', model=model_name, tokenizer=model_name)

def get_pdf_text(pdf_file: str) -> [str] :
    with open(pdf_file, 'rb') as pdf:
        reader = PyPDF2.PdfReader(pdf, strict = False)
        pdf_text = ''

        for page in reader.pages :
            text = page.extract_text()
            pdf_text+=(' ' + text)

        return pdf_text


if __name__ == '__main__' :
    content = get_pdf_text('sample.pdf')
# print(content)

while True :
    question = str(input('Ask a question from the given pdf: '))
    prompt = {
        'question': question,
        'context': content
    }
    result = genie(prompt)
    print(result['answer'])