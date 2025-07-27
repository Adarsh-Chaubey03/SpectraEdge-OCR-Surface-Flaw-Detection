#!/usr/bin/env python3
"""
Launcher script for Flipkart Grid OCR & Surface Flaw Detection System
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'opencv-python',
        'pytesseract', 
        'flask',
        'numpy',
        'scipy',
        'scikit-image',
        'easyocr',
        'reportlab',
        'pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Installing missing packages...")
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ All packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please run:")
            print(f"   pip install {' '.join(missing_packages)}")
            return False
    
    return True

def check_tesseract():
    """Check if Tesseract OCR is installed"""
    import pytesseract
    
    try:
        # Try to get Tesseract version
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract OCR found (version: {version})")
        return True
    except Exception as e:
        print("❌ Tesseract OCR not found or not properly configured")
        print("📥 Please install Tesseract OCR from:")
        print("   https://github.com/UB-Mannheim/tesseract/wiki")
        print("\nAfter installation, update the path in config.py if needed")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'captured_images',
        'reports', 
        'logs',
        'uploads',
        'utils'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Directories created successfully")

def check_camera():
    """Check if camera is available"""
    import cv2
    
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✅ Camera is working properly")
                return True
            else:
                print("⚠️  Camera found but cannot capture frames")
                return False
        else:
            print("❌ No camera found")
            return False
    except Exception as e:
        print(f"❌ Camera error: {e}")
        return False

def main():
    """Main launcher function"""
    print("🚀 Flipkart Grid OCR & Surface Flaw Detection System")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        return
    
    # Check Tesseract
    print("\n🔍 Checking Tesseract OCR...")
    if not check_tesseract():
        return
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Check camera
    print("\n📷 Checking camera...")
    camera_ok = check_camera()
    if not camera_ok:
        print("⚠️  Camera issues detected. The system may not work properly.")
    
    # Start the application
    print("\n🎯 Starting the application...")
    print("🌐 The web interface will be available at: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 60)
    
    try:
        # Import and run the main application
        from app import app
        
        print("🚀 System started successfully!")
        print("📱 Access the application at: http://localhost:5000")
        print("🔍 Features available:")
        print("   - Real-time OCR with multiple engines")
        print("   - Advanced surface defect detection")
        print("   - Image quality analysis")
        print("   - Flipkart product comparison")
        print("   - Comprehensive PDF reporting")
        print("\n💡 You can also run directly with: python app.py")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n⏹️  Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        print("Please check the error message above and try again")

if __name__ == "__main__":
    main() 