import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


def extract_symbols(image, threshold=128):
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    symbols = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        symbol = binary_image[y:y + h, x:x + w]
        symbols.append((symbol, (x, y, w, h)))

    symbols = sorted(symbols, key=lambda s: s[1][0])
    return symbols


def preprocess_symbol(symbol, size=(30, 30)):
    resized = cv2.resize(symbol, size, interpolation=cv2.INTER_AREA)
    normalized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
    return normalized


def extract_features(symbol):
    moments = cv2.moments(symbol)
    hu_moments = cv2.HuMoments(moments).flatten()
    log_features = -np.sign(hu_moments) * np.log(np.abs(hu_moments) + 1e-7)
    return log_features


def create_templates(template_image):
    symbols_with_positions = extract_symbols(template_image)
    alphabet = "AB01WX*-/PD"


    templates = {}
    for (symbol, _), char in zip(symbols_with_positions, alphabet):
        preprocessed = preprocess_symbol(symbol)
        features = extract_features(preprocessed)
        templates[char] = features

    return templates


def recognize_symbol(symbol_image, templates):
    symbol_features = extract_features(preprocess_symbol(symbol_image))

    min_distance = float('inf')
    recognized_char = None

    for char, template_features in templates.items():
        distance = np.linalg.norm(symbol_features - template_features)
        if distance < min_distance:
            min_distance = distance
            recognized_char = char

    return recognized_char


def create_frequency_dictionary(symbols_image, templates):
    symbols = extract_symbols(symbols_image)
    frequency_dict = defaultdict(int)

    for symbol, _ in symbols:
        recognized_char = recognize_symbol(symbol, templates)
        if recognized_char is not None:
            frequency_dict[recognized_char] += 1

    return frequency_dict


if __name__ == "__main__":
    alphabet_image = cv2.imread("alphabet_ext.png", cv2.IMREAD_GRAYSCALE)
    symbols_image = cv2.imread("symbols.png", cv2.IMREAD_GRAYSCALE)

    templates = create_templates(alphabet_image)
    freq_dict = create_frequency_dictionary(symbols_image, templates)

    print("Частотный словарь символов:")
    for char, count in sorted(freq_dict.items()):
        print(f"'{char}': {count}")

    plt.imshow(symbols_image, cmap="gray")
    plt.title("Распознанные символы")
    plt.show()
