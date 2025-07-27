#!/usr/bin/env python3
"""
Test script for Flipkart Grid OCR & Surface Flaw Detection System
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

def test_opencv():
    """Test OpenCV functionality"""
    print("🔍 Testing OpenCV...")
    try:
        # Test basic OpenCV operations
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:] = (255, 0, 0)  # Blue color
        
        # Test color conversion
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        # Test blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Test edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        print("✅ OpenCV test passed")
        return True
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def test_numpy():
    """Test NumPy functionality"""
    print("🔍 Testing NumPy...")
    try:
        # Test basic NumPy operations
        arr = np.array([1, 2, 3, 4, 5])
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        
        # Test array operations
        zeros = np.zeros((10, 10))
        ones = np.ones((10, 10))
        
        print("✅ NumPy test passed")
        return True
    except Exception as e:
        print(f"❌ NumPy test failed: {e}")
        return False

def test_pytesseract():
    """Test Tesseract OCR"""
    print("🔍 Testing Tesseract OCR...")
    try:
        import pytesseract
        
        # Create a simple test image with text
        test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        
        # Add text to image
        cv2.putText(test_image, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        
        # Convert to grayscale
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        # Try OCR
        text = pytesseract.image_to_string(gray)
        
        print("✅ Tesseract OCR test passed")
        return True
    except Exception as e:
        print(f"❌ Tesseract OCR test failed: {e}")
        return False

def test_easyocr():
    """Test EasyOCR"""
    print("🔍 Testing EasyOCR...")
    try:
        import easyocr
        
        # Initialize reader
        reader = easyocr.Reader(['en'])
        
        # Create a simple test image
        test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        
        # Try OCR
        results = reader.readtext(test_image)
        
        print("✅ EasyOCR test passed")
        return True
    except Exception as e:
        print(f"❌ EasyOCR test failed: {e}")
        return False

def test_flask():
    """Test Flask"""
    print("🔍 Testing Flask...")
    try:
        from flask import Flask
        
        app = Flask(__name__)
        
        @app.route('/test')
        def test():
            return "Flask is working!"
        
        print("✅ Flask test passed")
        return True
    except Exception as e:
        print(f"❌ Flask test failed: {e}")
        return False

def test_reportlab():
    """Test ReportLab"""
    print("🔍 Testing ReportLab...")
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        # Create a test PDF
        test_pdf_path = "test_report.pdf"
        c = canvas.Canvas(test_pdf_path, pagesize=letter)
        c.drawString(100, 750, "Test Report")
        c.save()
        
        # Clean up
        if os.path.exists(test_pdf_path):
            os.remove(test_pdf_path)
        
        print("✅ ReportLab test passed")
        return True
    except Exception as e:
        print(f"❌ ReportLab test failed: {e}")
        return False

def test_camera():
    """Test camera functionality"""
    print("🔍 Testing camera...")
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("⚠️  No camera available")
            return False
        
        # Try to capture a frame
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            print(f"✅ Camera test passed (Frame size: {frame.shape})")
            return True
        else:
            print("❌ Camera test failed - cannot capture frames")
            return False
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_defect_detection():
    """Test defect detection algorithms"""
    print("🔍 Testing defect detection...")
    try:
        # Create a test image with simulated defects
        test_image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        
        # Add a simulated scratch (line)
        cv2.line(test_image, (50, 100), (150, 100), (0, 0, 0), 3)
        
        # Add a simulated dent (circle)
        cv2.circle(test_image, (100, 50), 20, (0, 0, 0), -1)
        
        # Test basic image processing
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"✅ Defect detection test passed (Found {len(contours)} contours)")
        return True
    except Exception as e:
        print(f"❌ Defect detection test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("🔍 Testing configuration...")
    try:
        from config import Config
        
        # Test basic configuration
        config = Config()
        
        # Check if required attributes exist
        required_attrs = ['CAMERA_INDEX', 'CAMERA_WIDTH', 'CAMERA_HEIGHT', 'FLIPKART_PRODUCTS']
        
        for attr in required_attrs:
            if hasattr(config, attr):
                print(f"   ✅ {attr}: {getattr(config, attr)}")
            else:
                print(f"   ❌ Missing {attr}")
                return False
        
        print("✅ Configuration test passed")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def create_test_image():
    """Create a test image for OCR testing"""
    print("🔍 Creating test image...")
    try:
        # Create a test image with product information
        test_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Add product text
        cv2.putText(test_image, "Samsung Galaxy S21", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        cv2.putText(test_image, "Price: Rs. 45,999", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(test_image, "Flipkart", (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Add simulated defects
        cv2.line(test_image, (100, 300), (200, 300), (0, 0, 255), 3)  # Scratch
        cv2.circle(test_image, (400, 100), 15, (0, 255, 0), -1)  # Dent
        
        # Save test image
        cv2.imwrite("test_product_image.jpg", test_image)
        print("✅ Test image created: test_product_image.jpg")
        return True
    except Exception as e:
        print(f"❌ Test image creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Flipkart Grid OCR & Surface Flaw Detection System - Test Suite")
    print("=" * 70)
    
    tests = [
        ("OpenCV", test_opencv),
        ("NumPy", test_numpy),
        ("Tesseract OCR", test_pytesseract),
        ("EasyOCR", test_easyocr),
        ("Flask", test_flask),
        ("ReportLab", test_reportlab),
        ("Camera", test_camera),
        ("Defect Detection", test_defect_detection),
        ("Configuration", test_configuration),
        ("Test Image Creation", create_test_image)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        print()
    
    print("=" * 70)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\n🚀 To start the system, run:")
        print("   python app.py")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("\n🔧 Common solutions:")
        print("   1. Install missing packages: pip install -r requirements.txt")
        print("   2. Install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   3. Check camera connection")
        print("   4. Update Tesseract path in config.py if needed")

if __name__ == "__main__":
    main() 