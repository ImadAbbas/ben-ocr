# coding: utf-8

import os
import operator
import contextlib
import zipfile
import fitz
import requests
import ujson
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from invoke import task
from bookworm import typehints as t
from bookworm.logger import logger
from bookworm.i18n import LocaleInfo
from bookworm.ocr_engines import OcrRequest
from bookworm.ocr_engines.tesseract_ocr_engine import TesseractOcrEngine
from bookworm.image_io import ImageIO


log = logger.getChild("ben_ocr")


HERE = Path.cwd()
PDF_FILES_DIR = HERE / "pdf_files"
JSON_OUTPUT_DIR = HERE / "ocr_output_json"
# Change as required based on scan quality
PAGE_IMAGE_ZOOM_MATRIX = fitz.Matrix(2, 2)  # 2x
# Change as required based on recognition languages
# Multiple languages are used if the text is multilingual
# Remove the additional languages if not needed
RECOGNITION_LANGUAGES = [
    LocaleInfo.from_three_letter_code("ara"),  # Arabic
    #    LocaleInfo.from_three_letter_code("eng"), # English
]


@task
def download_tesseract(c):
    tesseract_download_url = f"https://raw.githubusercontent.com/blindpandas/bookworm/develop/packages/tesseract/tesseract_x64.zip"
    log.info(f"Downloading tesseract from {tesseract_download_url}")
    archive_data = requests.get(tesseract_download_url).content
    archive_file = BytesIO(archive_data)
    log.info("Downloaded tesseract OCR engine")
    tesseract_directory = HERE / "tesseract_ocr"
    tesseract_directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_file, "r") as zfile:
        zfile.extractall(path=tesseract_directory)
    # Download Arabic Language
    log.info("Downloading Arabic Language")
    ara_best_traineddata_download_url = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/ara.traineddata"
    ara_traineddata = requests.get(ara_best_traineddata_download_url).content
    ara_file = tesseract_directory / "tessdata" / "ara.traineddata"
    ara_file.write_bytes(ara_traineddata)
    log.info("Done downloading Arabic Language")


@task
def archive_json_output(c):
    log.info("Creating output archive...")
    with zipfile.ZipFile("OCR_output.zip", "w") as archive:
        for filename in JSON_OUTPUT_DIR.glob("*.json"):
            archive.write(filename, filename.name)
    log.info("Created archive: OCR_output.zip ")


@task(
    pre=[
        download_tesseract,
    ],
    post=[
        archive_json_output,
    ],
)
def ocr(c):
    log.info("Starting BEN OCR tasks...")
    for pdf_filename in PDF_FILES_DIR.glob("*.pdf"):
        perform_ocr_on_pdf(pdf_filename)
    log.info("Done processing all PDF files")


def image_to_text(image: ImageIO, languages: list[LocaleInfo], cookie: t.Any) -> str:
    tesseract = TesseractOcrEngine()
    tesseract.check()
    ocr_request = OcrRequest(
        languages=languages, image=image, image_processing_pipelines=(), cookie=cookie
    )
    return tesseract.recognize(ocr_request)


def pdf_page_to_image(pdf_page: fitz.Page) -> ImageIO:
    pix = pdf_page.get_pixmap(matrix=PAGE_IMAGE_ZOOM_MATRIX, alpha=False)
    return ImageIO(data=pix.samples, width=pix.width, height=pix.height)


def _ocr_pdf_page(args):
    page_index, page = args
    image = pdf_page_to_image(page)
    return image_to_text(image, RECOGNITION_LANGUAGES, cookie=page_index)


def perform_ocr_on_pdf(pdf_filename: t.PathLike):
    log.info(f"OCR Scanning file: {pdf_filename.name}")
    pdf = fitz.open(pdf_filename)
    recog_results = []
    with contextlib.closing(pdf):
        with ThreadPoolExecutor(thread_name_prefix="ben.ocr") as executor:
            work_units = enumerate(pdf)
            for recognition_result in executor.map(_ocr_pdf_page, work_units):
                log.info(f"Processed one page of file: {pdf_filename.name}")
                recog_results.append(recognition_result)
    log.info(f"Finished processing one page of file: {pdf_filename.name}")
    # Sort pages
    recog_results.sort(key=operator.attrgetter("cookie"))
    # JSON object
    data = {
        "title": pdf_filename.stem,
        "content": [res.recognized_text for res in recog_results],
        "chapters": [],
    }
    json_file = JSON_OUTPUT_DIR / f"{pdf_filename.stem}.json"
    json_string = ujson.dumps(data, ensure_ascii=False, indent=2)
    json_file.write_text(json_string, encoding="utf-8")
    log.info(f"Wrote json output to file: {json_file.name}")
