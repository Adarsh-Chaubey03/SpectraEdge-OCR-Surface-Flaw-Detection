# Configuration file for OCR and Surface Flaw Detection System

import os

class Config:
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    DEBUG = True
    
    # Camera Configuration
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 1920
    CAMERA_HEIGHT = 1080
    CAMERA_FPS = 30
    
    # OCR Configuration
    TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    OCR_LANGUAGES = ['en']
    OCR_CONFIG = '--psm 6 --oem 3'
    
    # Image Processing Configuration
    IMAGE_QUALITY_THRESHOLD = 70.0
    MIN_DEFECT_AREA = 100
    MAX_DEFECT_AREA = 10000
    
    # Surface Defect Detection Configuration
    DEFECT_DETECTION_CONFIG = {
        'scratch': {
            'threshold': 0.3,
            'color': (0, 0, 255),
            'min_area': 50,
            'max_area': 1000
        },
        'dent': {
            'threshold': 0.4,
            'color': (0, 255, 0),
            'min_area': 200,
            'max_area': 5000
        },
        'crack': {
            'threshold': 0.5,
            'color': (255, 0, 0),
            'min_area': 100,
            'max_area': 3000
        },
        'stain': {
            'threshold': 0.2,
            'color': (255, 255, 0),
            'min_area': 100,
            'max_area': 2000
        },
        'discoloration': {
            'threshold': 0.25,
            'color': (255, 0, 255),
            'min_area': 500,
            'max_area': 5000
        }
    }
    
    # Flipkart Product Database
    FLIPKART_PRODUCTS = {
        'samsung': {
            'name': 'Samsung Galaxy Smartphone',
            'category': 'Electronics',
            'expected_price_range': (15000, 50000),
            'common_defects': ['screen_scratch', 'body_damage', 'packaging_damage'],
            'quality_standards': {
                'min_image_quality': 80,
                'max_defect_count': 2,
                'critical_defects': ['screen_crack', 'water_damage']
            }
        },
        'apple': {
            'name': 'Apple iPhone',
            'category': 'Electronics',
            'expected_price_range': (40000, 150000),
            'common_defects': ['screen_crack', 'body_dent', 'packaging_tear'],
            'quality_standards': {
                'min_image_quality': 85,
                'max_defect_count': 1,
                'critical_defects': ['screen_crack', 'battery_damage']
            }
        },
        'nike': {
            'name': 'Nike Sports Shoes',
            'category': 'Footwear',
            'expected_price_range': (2000, 8000),
            'common_defects': ['sole_damage', 'fabric_tear', 'color_fading'],
            'quality_standards': {
                'min_image_quality': 75,
                'max_defect_count': 3,
                'critical_defects': ['sole_detachment', 'major_fabric_damage']
            }
        },
        'coca-cola': {
            'name': 'Coca-Cola Beverage',
            'category': 'Beverages',
            'expected_price_range': (20, 100),
            'common_defects': ['label_damage', 'bottle_crack', 'expiry_date'],
            'quality_standards': {
                'min_image_quality': 70,
                'max_defect_count': 2,
                'critical_defects': ['bottle_crack', 'expired_product']
            }
        },
        'pepsi': {
            'name': 'Pepsi Beverage',
            'category': 'Beverages',
            'expected_price_range': (20, 100),
            'common_defects': ['label_damage', 'bottle_crack', 'expiry_date'],
            'quality_standards': {
                'min_image_quality': 70,
                'max_defect_count': 2,
                'critical_defects': ['bottle_crack', 'expired_product']
            }
        },
        'nestle': {
            'name': 'Nestle Products',
            'category': 'Food & Beverages',
            'expected_price_range': (50, 500),
            'common_defects': ['packaging_damage', 'expiry_date', 'seal_break'],
            'quality_standards': {
                'min_image_quality': 75,
                'max_defect_count': 2,
                'critical_defects': ['expired_product', 'seal_break']
            }
        }
    }
    
    # File Paths
    UPLOAD_FOLDER = 'uploads'
    CAPTURED_IMAGES_FOLDER = 'captured_images'
    REPORTS_FOLDER = 'reports'
    LOGS_FOLDER = 'logs'
    
    # Report Configuration
    REPORT_CONFIG = {
        'company_name': 'Flipkart Grid',
        'system_name': 'OCR & Surface Defect Detection System',
        'report_template': 'comprehensive',
        'include_charts': True,
        'include_recommendations': True
    }
    
    # API Configuration
    API_CONFIG = {
        'rate_limit': 100,  # requests per minute
        'timeout': 30,  # seconds
        'max_file_size': 10 * 1024 * 1024  # 10MB
    }
    
    # Logging Configuration
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': 'logs/app.log'
    }
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        # Create necessary directories
        for folder in [Config.UPLOAD_FOLDER, Config.CAPTURED_IMAGES_FOLDER, 
                      Config.REPORTS_FOLDER, Config.LOGS_FOLDER]:
            os.makedirs(folder, exist_ok=True)

class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    
    # Production-specific settings
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 15

class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    
    # Testing-specific settings
    CAMERA_INDEX = 1  # Use different camera for testing

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 