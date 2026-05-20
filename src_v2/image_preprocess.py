import cv2


def resize_for_speed(image, max_width: int = 1200):
    """
    Resize image jika terlalu besar.

    Tujuan:
    - OCR lebih cepat
    - layout tetap proporsional
    """
    h, w = image.shape[:2]

    if w <= max_width:
        return image, 1.0

    scale = max_width / w

    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    return resized, scale


def light_preprocess(image):
    """
    Preprocessing ringan.

    V2 sengaja tidak selalu adaptive threshold karena pada beberapa mobile receipt
    warna/font tipis justru bisa hilang.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CLAHE ringan untuk menaikkan kontras lokal
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return enhanced