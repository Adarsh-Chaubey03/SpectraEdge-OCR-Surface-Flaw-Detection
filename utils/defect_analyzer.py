"""
Advanced Surface Defect Analysis Utilities
Provides sophisticated algorithms for detecting various types of surface defects
"""

import cv2
import numpy as np
from skimage import measure, morphology, filters, feature
from skimage.morphology import disk, square
from skimage.filters import threshold_otsu, gaussian
from scipy import ndimage
from scipy.stats import linregress
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AdvancedDefectAnalyzer:
    """Advanced surface defect detection and analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.defect_types = config.get('DEFECT_DETECTION_CONFIG', {})
        
    def analyze_surface_comprehensive(self, image: np.ndarray) -> Dict:
        """
        Comprehensive surface analysis using multiple algorithms
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary containing all analysis results
        """
        results = {
            'defects': [],
            'surface_quality': {},
            'recommendations': [],
            'confidence_scores': {}
        }
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Perform different types of analysis
        results['defects'].extend(self._detect_structural_defects(gray))
        results['defects'].extend(self._detect_color_defects(hsv, lab))
        results['defects'].extend(self._detect_texture_defects(gray))
        results['defects'].extend(self._detect_edge_defects(gray))
        
        # Analyze surface quality
        results['surface_quality'] = self._analyze_surface_quality(gray, hsv, lab)
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        # Calculate confidence scores
        results['confidence_scores'] = self._calculate_confidence_scores(results)
        
        return results
    
    def _detect_structural_defects(self, gray: np.ndarray) -> List[Dict]:
        """Detect structural defects like scratches, dents, cracks"""
        defects = []
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # Detect lines (potential scratches)
        lines = cv2.HoughLinesP(morph, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if length > 50:  # Minimum scratch length
                    defects.append({
                        'type': 'scratch',
                        'position': ((x1+x2)//2, (y1+y2)//2),
                        'dimensions': (length, 2),
                        'area': length * 2,
                        'confidence': min(0.9, length / 200),
                        'severity': 'medium' if length < 100 else 'high'
                    })
        
        # Detect circular/elliptical defects (dents)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=10, maxRadius=100)
        
        if circles is not None:
            for circle in circles[0]:
                x, y, r = circle
                defects.append({
                    'type': 'dent',
                    'position': (int(x), int(y)),
                    'dimensions': (int(r*2), int(r*2)),
                    'area': np.pi * r**2,
                    'confidence': min(0.8, r / 50),
                    'severity': 'medium' if r < 30 else 'high'
                })
        
        return defects
    
    def _detect_color_defects(self, hsv: np.ndarray, lab: np.ndarray) -> List[Dict]:
        """Detect color-based defects like stains and discolorations"""
        defects = []
        
        # Analyze saturation channel for stains
        saturation = hsv[:, :, 1]
        saturation_thresh = threshold_otsu(saturation)
        stain_mask = saturation < saturation_thresh
        
        # Find connected components
        labeled_stains = measure.label(stain_mask)
        regions = measure.regionprops(labeled_stains)
        
        for region in regions:
            if region.area > 100 and region.area < 5000:
                defects.append({
                    'type': 'stain',
                    'position': region.centroid[::-1],  # Convert to (x, y)
                    'dimensions': (region.bbox[3] - region.bbox[1], 
                                 region.bbox[2] - region.bbox[0]),
                    'area': region.area,
                    'confidence': min(0.7, region.area / 1000),
                    'severity': 'low' if region.area < 500 else 'medium'
                })
        
        # Analyze L channel for discolorations
        l_channel = lab[:, :, 0]
        l_std = np.std(l_channel)
        
        if l_std > 30:  # High variation indicates discoloration
            # Find regions with significant L variation
            l_gradient = np.gradient(l_channel)
            l_magnitude = np.sqrt(l_gradient[0]**2 + l_gradient[1]**2)
            
            discoloration_mask = l_magnitude > np.percentile(l_magnitude, 90)
            labeled_discolorations = measure.label(discoloration_mask)
            regions = measure.regionprops(labeled_discolorations)
            
            for region in regions:
                if region.area > 200:
                    defects.append({
                        'type': 'discoloration',
                        'position': region.centroid[::-1],
                        'dimensions': (region.bbox[3] - region.bbox[1], 
                                     region.bbox[2] - region.bbox[0]),
                        'area': region.area,
                        'confidence': min(0.6, region.area / 2000),
                        'severity': 'medium'
                    })
        
        return defects
    
    def _detect_texture_defects(self, gray: np.ndarray) -> List[Dict]:
        """Detect texture-based defects using Gabor filters and LBP"""
        defects = []
        
        # Apply Gabor filters for texture analysis
        angles = [0, 45, 90, 135]
        frequencies = [0.1, 0.2, 0.3]
        
        texture_responses = []
        for angle in angles:
            for freq in frequencies:
                kernel = cv2.getGaborKernel((21, 21), 5, np.radians(angle), 
                                          2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
                response = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                texture_responses.append(response)
        
        # Calculate texture variance
        texture_variance = np.var(texture_responses, axis=0)
        
        # Find regions with abnormal texture
        texture_thresh = np.percentile(texture_variance, 95)
        texture_mask = texture_variance > texture_thresh
        
        # Find connected components
        labeled_texture = measure.label(texture_mask)
        regions = measure.regionprops(labeled_texture)
        
        for region in regions:
            if region.area > 150:
                defects.append({
                    'type': 'texture_anomaly',
                    'position': region.centroid[::-1],
                    'dimensions': (region.bbox[3] - region.bbox[1], 
                                 region.bbox[2] - region.bbox[0]),
                    'area': region.area,
                    'confidence': min(0.5, region.area / 1000),
                    'severity': 'low'
                })
        
        return defects
    
    def _detect_edge_defects(self, gray: np.ndarray) -> List[Dict]:
        """Detect edge-based defects like cracks and tears"""
        defects = []
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Apply morphological operations to connect broken edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200 and area < 3000:
                # Calculate contour properties
                perimeter = cv2.arcLength(contour, True)
                x, y, w, h = cv2.boundingRect(contour)
                
                # Analyze shape characteristics
                aspect_ratio = float(w) / h if h > 0 else 0
                extent = float(area) / (w * h) if w * h > 0 else 0
                
                # Classify based on shape
                if aspect_ratio > 3 and extent < 0.3:
                    defect_type = 'crack'
                    confidence = min(0.8, perimeter / 200)
                elif aspect_ratio < 0.3 and extent < 0.3:
                    defect_type = 'tear'
                    confidence = min(0.7, area / 1000)
                else:
                    defect_type = 'edge_defect'
                    confidence = min(0.6, area / 500)
                
                defects.append({
                    'type': defect_type,
                    'position': (x + w//2, y + h//2),
                    'dimensions': (w, h),
                    'area': area,
                    'confidence': confidence,
                    'severity': 'high' if confidence > 0.7 else 'medium'
                })
        
        return defects
    
    def _analyze_surface_quality(self, gray: np.ndarray, hsv: np.ndarray, lab: np.ndarray) -> Dict:
        """Analyze overall surface quality metrics"""
        quality_metrics = {}
        
        # Sharpness analysis
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        quality_metrics['sharpness'] = min(100, laplacian_var * 2)
        
        # Brightness analysis
        brightness = np.mean(gray)
        quality_metrics['brightness'] = (brightness / 255) * 100
        
        # Contrast analysis
        contrast = np.std(gray)
        quality_metrics['contrast'] = (contrast / 255) * 100
        
        # Noise analysis
        noise = self._calculate_noise_level(gray)
        quality_metrics['noise_level'] = noise * 100
        
        # Color consistency
        color_std = np.std(hsv[:, :, 1])  # Saturation standard deviation
        quality_metrics['color_consistency'] = max(0, 100 - color_std)
        
        # Overall quality score
        weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # Sharpness, brightness, contrast, noise, color
        scores = [quality_metrics['sharpness'], quality_metrics['brightness'], 
                 quality_metrics['contrast'], 100 - quality_metrics['noise_level'], 
                 quality_metrics['color_consistency']]
        
        quality_metrics['overall_score'] = sum(w * s for w, s in zip(weights, scores))
        
        return quality_metrics
    
    def _calculate_noise_level(self, gray: np.ndarray) -> float:
        """Calculate noise level in the image"""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Calculate difference
        diff = cv2.absdiff(gray, blurred)
        
        # Return normalized noise level
        return np.mean(diff) / 255
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        defects = results['defects']
        quality = results['surface_quality']
        
        # Analyze defect severity
        high_severity_defects = [d for d in defects if d['severity'] == 'high']
        medium_severity_defects = [d for d in defects if d['severity'] == 'medium']
        
        if high_severity_defects:
            recommendations.append("⚠️ Critical defects detected - Immediate attention required")
        
        if medium_severity_defects:
            recommendations.append("⚠️ Moderate defects detected - Quality review recommended")
        
        # Quality-based recommendations
        if quality['overall_score'] < 70:
            recommendations.append("📸 Image quality below threshold - Improve lighting conditions")
        
        if quality['sharpness'] < 60:
            recommendations.append("🔍 Image appears blurry - Adjust camera focus")
        
        if quality['noise_level'] > 30:
            recommendations.append("📊 High noise detected - Reduce camera ISO or improve lighting")
        
        # Defect type specific recommendations
        defect_types = [d['type'] for d in defects]
        
        if 'scratch' in defect_types:
            recommendations.append("🔧 Surface scratches detected - Consider protective packaging")
        
        if 'crack' in defect_types:
            recommendations.append("🚨 Structural cracks detected - Product may be compromised")
        
        if 'stain' in defect_types:
            recommendations.append("🧽 Surface stains detected - Cleaning may be required")
        
        if not recommendations:
            recommendations.append("✅ Surface quality appears acceptable")
        
        return recommendations
    
    def _calculate_confidence_scores(self, results: Dict) -> Dict:
        """Calculate confidence scores for the analysis"""
        confidence_scores = {}
        
        defects = results['defects']
        
        if defects:
            # Calculate average confidence for defects
            avg_defect_confidence = np.mean([d['confidence'] for d in defects])
            confidence_scores['defect_detection'] = avg_defect_confidence
            
            # Calculate confidence based on defect count and types
            defect_count = len(defects)
            unique_types = len(set(d['type'] for d in defects))
            
            # More defects and types increase confidence
            confidence_scores['analysis_completeness'] = min(1.0, 
                (defect_count * 0.1 + unique_types * 0.2))
        else:
            confidence_scores['defect_detection'] = 0.0
            confidence_scores['analysis_completeness'] = 0.5
        
        # Quality analysis confidence
        quality = results['surface_quality']
        quality_confidence = quality['overall_score'] / 100
        confidence_scores['quality_analysis'] = quality_confidence
        
        # Overall confidence
        confidence_scores['overall'] = np.mean([
            confidence_scores['defect_detection'],
            confidence_scores['analysis_completeness'],
            confidence_scores['quality_analysis']
        ])
        
        return confidence_scores

class DefectVisualizer:
    """Utility class for visualizing defect detection results"""
    
    @staticmethod
    def draw_defects_on_image(image: np.ndarray, defects: List[Dict]) -> np.ndarray:
        """Draw detected defects on the image"""
        result_image = image.copy()
        
        # Color mapping for different defect types
        color_map = {
            'scratch': (0, 0, 255),      # Red
            'dent': (0, 255, 0),         # Green
            'crack': (255, 0, 0),        # Blue
            'stain': (255, 255, 0),      # Cyan
            'discoloration': (255, 0, 255), # Magenta
            'texture_anomaly': (0, 255, 255), # Yellow
            'edge_defect': (128, 128, 128),   # Gray
            'tear': (165, 42, 42)        # Brown
        }
        
        for defect in defects:
            defect_type = defect['type']
            color = color_map.get(defect_type, (0, 0, 255))
            
            x, y = defect['position']
            w, h = defect['dimensions']
            
            # Draw bounding box
            cv2.rectangle(result_image, 
                         (int(x - w//2), int(y - h//2)), 
                         (int(x + w//2), int(y + h//2)), 
                         color, 2)
            
            # Draw defect label
            label = f"{defect_type}: {defect['confidence']:.2f}"
            cv2.putText(result_image, label, 
                       (int(x - w//2), int(y - h//2) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result_image
    
    @staticmethod
    def create_analysis_report_image(image: np.ndarray, results: Dict) -> np.ndarray:
        """Create a comprehensive analysis report image"""
        # Create a larger canvas for the report
        height, width = image.shape[:2]
        report_height = height + 200
        report_image = np.ones((report_height, width, 3), dtype=np.uint8) * 255
        
        # Copy original image
        report_image[:height, :width] = image
        
        # Draw analysis results
        y_offset = height + 20
        
        # Title
        cv2.putText(report_image, "Surface Defect Analysis Report", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        y_offset += 40
        
        # Quality metrics
        quality = results.get('surface_quality', {})
        cv2.putText(report_image, f"Overall Quality: {quality.get('overall_score', 0):.1f}%", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_offset += 30
        
        # Defect summary
        defects = results.get('defects', [])
        cv2.putText(report_image, f"Defects Detected: {len(defects)}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_offset += 30
        
        # Confidence scores
        confidence = results.get('confidence_scores', {})
        cv2.putText(report_image, f"Analysis Confidence: {confidence.get('overall', 0):.2f}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return report_image 