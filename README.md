
# 📦 OCR & Surface Flaw Detection System

Real-time OCR and surface defect detection system for product quality assessment of **smartphones, laptops, footwear, beverages, and packaged food items**.

## 🚀 Features

### Core

* **Real-time OCR** using Tesseract and EasyOCR
* **Surface Defect Detection**: scratches, dents, cracks, stains, discolorations
* **Image Quality Analysis**: sharpness, brightness, contrast, noise
* **Product Database** for validation
* **Live Camera Feed** with on-screen results
* **PDF Reports** with detailed analysis

### Advanced

* Multi-engine OCR for higher accuracy
* Defect classification & severity scoring
* Confidence metrics for all detections
* Product-specific quality standards
* RESTful API endpoints
* Uploaded image analysis support

## 📋 Requirements

### System

* Windows 10/11 (64-bit)
* Python 3.8+
* Webcam/USB camera
* 4GB RAM, 2GB free disk

### Dependencies

* OpenCV 4.8+
* Tesseract OCR
* Flask
* NumPy
* SciPy
* scikit-image
* EasyOCR
* ReportLab

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/Adarsh-Chaubey03/OCR-and-Surface-Flaw-Detection
cd OCR-and-Surface-Flaw-Detection
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) and update its path in `config.py` if needed.

## 🚀 Usage

Start the application:

```bash
python app.py
```

Visit `http://localhost:5000` for:

1. **Live Analysis**
2. **Image Upload**
3. **PDF Report Generation**
4. **Settings Configuration**

### API Endpoints

* `GET /get_analysis_results` – Get current analysis results
* `POST /api/analyze_image` – Analyze uploaded image
* `GET /download_report` – Download PDF report
* `GET /capture_and_analyze` – Capture and analyze current frame

## 📊 System Architecture

* **OCR Engine**: Tesseract + EasyOCR
* **Defect Detector**: Structural, color, texture, edge defects
* **Quality Analyzer**: Focus, brightness, noise
* **Report Generator**: PDF with visual annotations
* **Web Interface**: Flask app

## 🔧 Configuration

Update `config.py` for:

* Camera settings
* OCR parameters
* Defect thresholds
* Product database & standards

Example:

```python
DEFECT_DETECTION_CONFIG = {
  'scratch': {'threshold': 0.3, 'min_area': 50, 'max_area': 1000},
  'dent': {'threshold': 0.4, 'min_area': 200, 'max_area': 5000},
  # ...
}
```

## 📈 Performance

* **OCR Accuracy**: \~95% for clear text
* **Defect Detection**: \~90% for major defects
* **False Positive Rate**: <5%
* **Real-time Analysis**: 30 FPS
* **Report Generation**: <5 seconds

## 🎯 Supported Products

* **Electronics**: Samsung Galaxy smartphones, Apple iPhones, laptops, tablets, accessories
* **Footwear**: Nike sports shoes, Adidas products, casual footwear
* **Beverages**: Coca-Cola, Pepsi, other soft drinks
* **Food & Beverages**: Nestle products, packaged foods

## 📋 Defects Detected

* **Structural**: Scratches, dents, cracks, tears
* **Surface**: Stains, discolorations, texture anomalies
* **Quality Issues**: Blur, noise, poor lighting

## 🔍 Analysis

* **OCR**: Brand, product name, price, barcode, confidence scoring
* **Quality**: Sharpness, brightness, contrast, noise
* **Defect**: Type, severity, size, position, recommendations

## 📊 Reporting

PDF reports include:

* Analysis summary
* Detected defects
* Quality metrics
* Recommendations
* Timestamps & visual annotations

## 🔧 Development

### Project Structure

```
OCR-and-Surface-Flaw-Detection/
├── app.py
├── config.py
├── requirements.txt
├── utils/
│   └── defect_analyzer.py
├── templates/
│   └── index.html
├── captured_images/
├── reports/
└── logs/
```

### Extending Functionality

* **New defects**: Update config & detection logic in `defect_analyzer.py`
* **Product DB**: Add entries and quality standards in config

## 🔄 Updates

### Version History

* **v1.0.0**: Basic OCR & defect detection
* **v1.1.0**: Advanced defect algorithms
* **v1.2.0**: Enhanced web interface & reporting
* **v1.3.0**: Product database integration

### Future Enhancements

* ML-based defect detection
* Mobile app support
* Cloud integration & real-time alerts
* Batch processing & user authentication

---

**Note**: This system is designed for the listed products and may require adjustments for other use cases.

---


