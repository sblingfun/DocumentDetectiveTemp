import os
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
from PIL import Image
import io
import requests
import tempfile
from tqdm.auto import tqdm
#from doctr.io import DocumentFile
#from doctr.models import ocr_predictor

#ocr_model = ocr_predictor(det_arch = 'db_resnet50',reco_arch = 'crnn_vgg16_bn', pretrained = True)

#ocr_model = ocr_predictor(pretrained = True)

class Pdf_path:
	def __init__(self, pdf_path):
	    self.pdf_path = pdf_path
#upload_attempts = 0

def get_pdf_path_name():
    upload_attempts = 0
    while upload_attempts < 4:
        upload_attempts += 1
        pdf_path = input("Enter the file path of the PDF attempt #" + str(upload_attempts) + "\n")
        if os.path.exists(pdf_path):
            print("File path exists")
            return Pdf_path(pdf_path)
            break


def text_formatter(text: str) -> str:
    #Formatting of text occurs here
    cleaned_text = text.replace("\n", " ").strip()

    return cleaned_text

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text = text)
        pages_and_texts.append({"page_number": page_number, #this setting is custom to nutrition pdf
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count_raw": len(text.split(". ")),
            "page_token_count": len(text) / 4,
            "text": text})
    return pages_and_texts
    
def open_and_read_pdf_ocr1(pdf_path: str) -> list[dict]:
    input_file = DocumentFile.from_pdf(pdf_path)
    ocr_result = ocr_model(input_file)
    extract_info = ocr_result.export()
    for obj1 in extract_info['pages'][0]["blocks"]:
        for obj2 in obj1["lines"]:
            for obj3 in obj2["words"]:
                print("{}: {}".format(obj3["geometry"],obj3["value"]))
    return extract_info
    
def download_tesseract_lang_data(lang):
    """Download Tesseract language data if not already present."""
    tessdata_dir = os.path.join(os.getenv('TESSDATA_PREFIX', ''), 'tessdata')
    if not os.path.exists(tessdata_dir):
        os.makedirs(tessdata_dir)

    lang_file = os.path.join(tessdata_dir, f'{lang}.traineddata')
    if not os.path.exists(lang_file):
        url = f'https://github.com/tesseract-ocr/tessdata_best/raw/main/{lang}.traineddata'
        r = requests.get(url)
        with open(lang_file, 'wb') as f:
            f.write(r.content)
            
def extract_text_and_images(input_pdf_path, output_pdf_path, dpi=300, lang='eng'):
    # Ensure language data is available
    for l in lang.split('+'):
        download_tesseract_lang_data(l)
        
def open_and_read_pdf_ocr_old(pdf_path: str) -> list[dict]:
    #download_tesseract_lang_data('eng')
    #print("download")
    #pdf = PdfReader(pdf_path)
    #print(len(pdf.pages))
    #num_pages = len(pdf.pages)
    #should free pdf from memory
    print("page Num")
    ocr_text_list = []
    
    #for page_number, page in tqdm(range(1, num_pages)):
        #image = 
    with tempfile.TemporaryDirectory() as path:
        images_from_path = convert_from_path(pdf_path, output_folder=path, dpi=300, fmt="jpeg")
    #images = convert_from_path(pdf_path, dpi=300)
        print("convert")
        #pages_and_texts = []
   
    for page_number, page in tqdm(enumerate(images_from_path)):
        ocr_text = pytesseract.image_to_string(page, lang='eng')
        del page
        #print("image_to_string")
            #print(ocr_text)
        ocr_text = text_formatter(text = ocr_text)
        ocr_text_list.append({"page_number": page_number,
            "page_char_count": len(ocr_text),
            "page_word_count": len(ocr_text.split(" ")),
            "page_sentence_count_raw": len(ocr_text.split(". ")),
            "page_token_count": len(ocr_text) / 4,
            "text": ocr_text})
        

            #print("append")
        #Need to figure out a way to combine 
        
        #pdf_page = fitz.open(pdf_path)[page_number]
        #print("fitz open")
        #pdf_text = pdf_page.get_text("text")
        #print("get_text")
        #pages_and_texts = pdf_text + "/n" + ocr_text
        #print("added " + pdf_text + "/n" + ocr_text)
    print("done ocr")
    
    return ocr_text_list
    
def open_and_read_pdf_ocr(pdf_path: str):
    pdf = PdfReader(pdf_path)
    num_pages = len(pdf.pages)
    ocr_text_list = []
    with tempfile.TemporaryDirectory() as path:
        for i in range(0, num_pages // 10):
            print("1st page: ", i*10 + 1)
            print("Lst page: ", (i+1)* 10)
            images_from_path = convert_from_path(pdf_path, output_folder=path, dpi=300, fmt="jpeg", first_page=i*10 + 1, last_page=min(num_pages,(i+1)*10))
            ocr_text_list.append({"sentences": next(perform_ocr_on_img(images_from_path))})
    return ocr_text_list
        
'''
            for image in images_from_path:
                ocr_text = pytesseract.image_to_string(image, lang='eng')
                ocr_text_list.append({"text": ocr_text})
            images_from_path = None
            '''
        
def open_and_ocr_pdf(pdf_path: str):
    ocr_text_list = []
    with fitz.open(pdf_path) as doc, tempfile.TemporaryDirectory() as path:
        mat = fitz.Matrix(1.2,1.2)
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            output = f'/{path}/{page.number}.jpg'
            pix.save(output)
            res = str(pytesseract.image_to_string(Image.open(output)))
            #print(res)
            ocr_text_list.append({"sentences":res,"page_number":page.number})
    return ocr_text_list


def perform_ocr_on_img(images) -> str:
    for image in images:
        yield pytesseract.image_to_string(image, lang='eng')

#def convert_pdf_to_images(pdf_path: str):
    

#pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)

#df = pd.DataFrame(pages_and_texts)
#print(df.head())

#print(df.describe().round(2))
