from paddleocr import PaddleOCR

ocr = PaddleOCR(
    device="gpu:0",
    lang="en"
    )

result = ocr.predict("./crypto_foia.pdf")
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
