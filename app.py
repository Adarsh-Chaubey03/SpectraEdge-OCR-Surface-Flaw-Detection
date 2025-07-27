import cv2
import numpy as np
import pytesseract
import easyocr
from flask import Flask, render_template, Response, send_file, jsonify, request
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import json
import os
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import measure, morphology, filters
from skimage.feature import canny
from skimage.morphology import disk
import pandas as pd
from collections import defaultdict
import threading
import time

app = Flask(__name__)

# Initialize camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
camera.set(cv2.CAP_PROP_FPS, 30)

# Initialize OCR readers
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
easy_reader = easyocr.Reader(['en'])

# Global variables for storing detection results
detection_results = {
    'brand_name': '',
    'product_name': '',
    'price': '',
    'barcode': '',
    'surface_defects': [],
    'confidence_scores': {},
    'timestamp': '',
    'image_quality': 0.0
}

# Flipkart product database (simulated)
flipkart_products = {
    'samsung': {
        'name': 'Samsung Galaxy Smartphone',
        'category': 'Electronics',
        'expected_price_range': (15000, 50000),
        'common_defects': ['screen_scratch', 'body_damage', 'packaging_damage']
    },
    'apple': {
        'name': 'Apple iPhone',
        'category': 'Electronics', 
        'expected_price_range': (40000, 150000),
        'common_defects': ['screen_crack', 'body_dent', 'packaging_tear']
    },
    'nike': {
        'name': 'Nike Sports Shoes',
        'category': 'Footwear',
        'expected_price_range': (2000, 8000),
        'common_defects': ['sole_damage', 'fabric_tear', 'color_fading']
    },
    'coca-cola': {
        'name': 'Coca-Cola Beverage',
        'category': 'Beverages',
        'expected_price_range': (20, 100),
        'common_defects': ['label_damage', 'bottle_crack', 'expiry_date']
    }
}

class SurfaceDefectDetector:
    def __init__(self):
        self.defect_types = {
            'scratch': {'threshold': 0.3, 'color': (0, 0, 255)},
            'dent': {'threshold': 0.4, 'color': (0, 255, 0)},
            'crack': {'threshold': 0.5, 'color': (255, 0, 0)},
            'stain': {'threshold': 0.2, 'color': (255, 255, 0)},
            'discoloration': {'threshold': 0.25, 'color': (255, 0, 255)}
        }
    
    def preprocess_image(self, image):
        """Enhance image for better defect detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(blurred)
        
        return equalized
    
    def detect_edges(self, image):
        """Detect edges using Canny algorithm"""
        edges = cv2.Canny(image, 50, 150)
        return edges
    
    def detect_contours(self, image):
        """Detect contours for defect analysis"""
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def analyze_surface_defects(self, image):
        """Comprehensive surface defect analysis"""
        defects = []
        
        # Preprocess image
        processed = self.preprocess_image(image)
        
        # Detect edges
        edges = self.detect_edges(processed)
        
        # Find contours
        contours = self.detect_contours(edges)
        
        # Analyze each contour for defects
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                # Calculate contour properties
                perimeter = cv2.arcLength(contour, True)
                x, y, w, h = cv2.boundingRect(contour)
                
                # Analyze shape characteristics
                aspect_ratio = float(w) / h if h > 0 else 0
                extent = float(area) / (w * h) if w * h > 0 else 0
                
                # Classify defect type based on characteristics
                defect_type = self.classify_defect(area, perimeter, aspect_ratio, extent)
                
                if defect_type:
                    defects.append({
                        'type': defect_type,
                        'area': area,
                        'position': (x, y),
                        'dimensions': (w, h),
                        'confidence': self.calculate_confidence(area, perimeter, aspect_ratio, extent)
                    })
        
        return defects
    
    def classify_defect(self, area, perimeter, aspect_ratio, extent):
        """Classify defect type based on geometric properties"""
        if aspect_ratio > 5 and area < 500:
            return 'scratch'
        elif extent < 0.3 and area > 1000:
            return 'dent'
        elif aspect_ratio < 0.2 and area > 2000:
            return 'crack'
        elif extent > 0.8 and area < 2000:
            return 'stain'
        elif 0.3 < extent < 0.7 and area > 500:
            return 'discoloration'
        return None
    
    def calculate_confidence(self, area, perimeter, aspect_ratio, extent):
        """Calculate confidence score for defect detection"""
        # Normalize parameters
        area_norm = min(area / 10000, 1.0)
        perimeter_norm = min(perimeter / 500, 1.0)
        
        # Calculate confidence based on multiple factors
        confidence = (area_norm * 0.4 + perimeter_norm * 0.3 + 
                     (1 - abs(aspect_ratio - 1)) * 0.2 + extent * 0.1)
        
        return min(confidence, 1.0)

class AdvancedOCR:
    def __init__(self):
        self.brand_keywords = [
            'samsung', 'apple', 'nike', 'adidas', 'coca-cola', 'pepsi', 
            'nestle', 'amazon', 'flipkart', 'myntra', 'zara', 'h&m'
        ]
        self.price_patterns = [
            r'₹\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
            r'Rs\.\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
            r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*₹',
            r'INR\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?'
        ]
    
    def enhance_image_for_ocr(self, image):
        """Enhance image for better OCR accuracy"""
        # Convert to PIL Image for enhancement
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.5)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(2.0)
        
        # Convert back to OpenCV format
        enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return enhanced
    
    def extract_text_with_multiple_methods(self, image):
        """Extract text using multiple OCR methods for better accuracy"""
        results = {}
        
        # Method 1: Tesseract OCR
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            tesseract_text = pytesseract.image_to_string(gray, config='--psm 6')
            results['tesseract'] = tesseract_text.strip()
        except Exception as e:
            results['tesseract'] = f"Error: {str(e)}"
        
        # Method 2: EasyOCR
        try:
            easyocr_results = easy_reader.readtext(image)
            easyocr_text = ' '.join([text[1] for text in easyocr_results])
            results['easyocr'] = easyocr_text.strip()
        except Exception as e:
            results['easyocr'] = f"Error: {str(e)}"
        
        # Method 3: Enhanced image with Tesseract
        try:
            enhanced = self.enhance_image_for_ocr(image)
            gray_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            enhanced_text = pytesseract.image_to_string(gray_enhanced, config='--psm 6')
            results['enhanced_tesseract'] = enhanced_text.strip()
        except Exception as e:
            results['enhanced_tesseract'] = f"Error: {str(e)}"
        
        return results
    
    def extract_brand_name(self, ocr_results):
        """Extract brand name from OCR results"""
        all_text = ' '.join(ocr_results.values()).lower()
        
        for brand in self.brand_keywords:
            if brand.lower() in all_text:
                return brand.title()
        
        return "Unknown Brand"
    
    def extract_price(self, ocr_results):
        """Extract price information from OCR results"""
        import re
        
        all_text = ' '.join(ocr_results.values())
        
        for pattern in self.price_patterns:
            matches = re.findall(pattern, all_text)
            if matches:
                return matches[0]
        
        return "Price not detected"
    
    def extract_product_name(self, ocr_results):
        """Extract product name from OCR results"""
        # This is a simplified version - in a real system, you'd use NLP
        all_text = ' '.join(ocr_results.values())
        
        # Look for common product keywords
        product_keywords = ['phone', 'laptop', 'shoes', 'shirt', 'bottle', 'book']
        
        for keyword in product_keywords:
            if keyword.lower() in all_text.lower():
                return f"Product containing '{keyword}'"
        
        return "Product name not detected"

class QualityAnalyzer:
    def __init__(self):
        self.quality_metrics = {}
    
    def analyze_image_quality(self, image):
        """Analyze overall image quality"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate brightness
        brightness = np.mean(gray)
        
        # Calculate contrast
        contrast = np.std(gray)
        
        # Calculate noise level
        noise = self.calculate_noise(gray)
        
        # Overall quality score (0-100)
        quality_score = min(100, (laplacian_var * 0.4 + 
                                 (brightness / 255) * 30 + 
                                 (contrast / 255) * 30 + 
                                 (1 - noise) * 20))
        
        return {
            'overall_score': quality_score,
            'sharpness': laplacian_var,
            'brightness': brightness,
            'contrast': contrast,
            'noise_level': noise
        }
    
    def calculate_noise(self, gray_image):
        """Calculate noise level in the image"""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Calculate difference
        diff = cv2.absdiff(gray_image, blurred)
        
        # Return noise level
        return np.mean(diff) / 255

# Initialize detection classes
defect_detector = SurfaceDefectDetector()
ocr_processor = AdvancedOCR()
quality_analyzer = QualityAnalyzer()

def generate_frames():
    """Generate video frames with real-time analysis"""
    global detection_results
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Create a copy for analysis
        analysis_frame = frame.copy()
        
        # Perform OCR analysis
        ocr_results = ocr_processor.extract_text_with_multiple_methods(analysis_frame)
        
        # Extract information
        brand_name = ocr_processor.extract_brand_name(ocr_results)
        product_name = ocr_processor.extract_product_name(ocr_results)
        price = ocr_processor.extract_price(ocr_results)
        
        # Perform surface defect analysis
        surface_defects = defect_detector.analyze_surface_defects(analysis_frame)
        
        # Analyze image quality
        quality_metrics = quality_analyzer.analyze_image_quality(analysis_frame)
        
        # Update global results
        detection_results.update({
            'brand_name': brand_name,
            'product_name': product_name,
            'price': price,
            'surface_defects': surface_defects,
            'confidence_scores': {
                'ocr_confidence': len([r for r in ocr_results.values() if 'Error' not in r]) / len(ocr_results),
                'defect_confidence': np.mean([d['confidence'] for d in surface_defects]) if surface_defects else 0
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'image_quality': quality_metrics['overall_score']
        })
        
        # Draw analysis results on frame
        frame = draw_analysis_results(frame, detection_results)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def draw_analysis_results(frame, results):
    """Draw analysis results on the video frame"""
    # Draw brand name
    cv2.putText(frame, f"Brand: {results['brand_name']}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw product name
    cv2.putText(frame, f"Product: {results['product_name']}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw price
    cv2.putText(frame, f"Price: {results['price']}", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw defect count
    defect_count = len(results['surface_defects'])
    cv2.putText(frame, f"Defects: {defect_count}", 
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                (0, 0, 255) if defect_count > 0 else (0, 255, 0), 2)
    
    # Draw quality score
    cv2.putText(frame, f"Quality: {results['image_quality']:.1f}%", 
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                (255, 0, 0) if results['image_quality'] < 70 else (0, 255, 0), 2)
    
    # Draw defects on frame
    for defect in results['surface_defects']:
        x, y = defect['position']
        w, h = defect['dimensions']
        color = defect_detector.defect_types.get(defect['type'], {}).get('color', (0, 0, 255))
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, defect['type'], (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_analysis_results')
def get_analysis_results():
    """API endpoint to get current analysis results"""
    return jsonify(detection_results)

@app.route('/capture_and_analyze')
def capture_and_analyze():
    """Capture current frame and perform detailed analysis"""
    success, frame = camera.read()
    if not success:
        return jsonify({'error': 'Failed to capture frame'})
    
    # Perform comprehensive analysis
    ocr_results = ocr_processor.extract_text_with_multiple_methods(frame)
    surface_defects = defect_detector.analyze_surface_defects(frame)
    quality_metrics = quality_analyzer.analyze_image_quality(frame)
    
    # Save captured image
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_path = f'captured_images/capture_{timestamp}.jpg'
    os.makedirs('captured_images', exist_ok=True)
    cv2.imwrite(image_path, frame)
    
    analysis_result = {
        'timestamp': datetime.now().isoformat(),
        'image_path': image_path,
        'ocr_results': ocr_results,
        'brand_name': ocr_processor.extract_brand_name(ocr_results),
        'product_name': ocr_processor.extract_product_name(ocr_results),
        'price': ocr_processor.extract_price(ocr_results),
        'surface_defects': surface_defects,
        'quality_metrics': quality_metrics,
        'flipkart_comparison': compare_with_flipkart_data(ocr_results)
    }
    
    return jsonify(analysis_result)

def compare_with_flipkart_data(ocr_results):
    """Compare detected product with Flipkart database"""
    all_text = ' '.join(ocr_results.values()).lower()
    
    for brand, data in flipkart_products.items():
        if brand.lower() in all_text:
            return {
                'matched_brand': brand,
                'expected_name': data['name'],
                'category': data['category'],
                'expected_price_range': data['expected_price_range'],
                'common_defects': data['common_defects']
            }
    
    return {'matched_brand': None}

@app.route('/download_report')
def download_report():
    """Generate and download comprehensive PDF report"""
    # Create PDF buffer
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1
    )
    story.append(Paragraph("Flipkart Grid OCR & Surface Defect Detection Report", title_style))
    story.append(Spacer(1, 20))
    
    # Current analysis results
    story.append(Paragraph("Current Analysis Results", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    # Create results table
    results_data = [
        ['Metric', 'Value'],
        ['Brand Name', detection_results['brand_name']],
        ['Product Name', detection_results['product_name']],
        ['Price', detection_results['price']],
        ['Image Quality', f"{detection_results['image_quality']:.1f}%"],
        ['Defect Count', str(len(detection_results['surface_defects']))],
        ['OCR Confidence', f"{detection_results['confidence_scores']['ocr_confidence']:.2f}"],
        ['Defect Confidence', f"{detection_results['confidence_scores']['defect_confidence']:.2f}"],
        ['Timestamp', detection_results['timestamp']]
    ]
    
    results_table = Table(results_data)
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(results_table)
    story.append(Spacer(1, 20))
    
    # Surface defects details
    if detection_results['surface_defects']:
        story.append(Paragraph("Surface Defects Detected", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        defects_data = [['Type', 'Area', 'Position', 'Confidence']]
        for defect in detection_results['surface_defects']:
            defects_data.append([
                defect['type'],
                f"{defect['area']:.0f}",
                f"({defect['position'][0]}, {defect['position'][1]})",
                f"{defect['confidence']:.2f}"
            ])
        
        defects_table = Table(defects_data)
        defects_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.red),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightcoral),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(defects_table)
    else:
        story.append(Paragraph("No Surface Defects Detected", styles['Heading2']))
        story.append(Paragraph("✓ Product surface appears to be in good condition", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"flipkart_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mimetype='application/pdf'
    )

@app.route('/api/analyze_image', methods=['POST'])
def analyze_uploaded_image():
    """API endpoint for analyzing uploaded images"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Read image
    image_bytes = file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({'error': 'Invalid image format'})
    
    # Perform analysis
    ocr_results = ocr_processor.extract_text_with_multiple_methods(image)
    surface_defects = defect_detector.analyze_surface_defects(image)
    quality_metrics = quality_analyzer.analyze_image_quality(image)
    
    analysis_result = {
        'ocr_results': ocr_results,
        'brand_name': ocr_processor.extract_brand_name(ocr_results),
        'product_name': ocr_processor.extract_product_name(ocr_results),
        'price': ocr_processor.extract_price(ocr_results),
        'surface_defects': surface_defects,
        'quality_metrics': quality_metrics,
        'flipkart_comparison': compare_with_flipkart_data(ocr_results)
    }
    
    return jsonify(analysis_result)

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('captured_images', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    print("🚀 Starting OCR & Surface Defect Detection System")
    print("📱 Access the application at: http://localhost:5000")
    print("🔍 Features:")
    print("   - Real-time OCR with multiple engines")
    print("   - Advanced surface defect detection")
    print("   - Image quality analysis")
    print("   - Flipkart product comparison")
    print("   - Comprehensive PDF reporting")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 