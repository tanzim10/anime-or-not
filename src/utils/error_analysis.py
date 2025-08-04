import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from collections import defaultdict
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

"""
ERROR ANALYSIS APPROACH GUIDE FOR ANIMEORNOT PROJECT
==================================================

This module provides a systematic approach to error analysis for image classification models.
Follow this step-by-step approach to understand model failures and improve performance.

APPROACH CHECKLIST:
==================

1. INITIAL ASSESSMENT
   □ Collect all predictions and ground truth labels
   □ Calculate overall accuracy, precision, recall, F1-score
   □ Identify total number of misclassified samples
   □ Determine if errors are systematic or random

2. CONFUSION MATRIX ANALYSIS
   □ Generate confusion matrix heatmap
   □ Identify most confused class pairs
   □ Look for systematic misclassification patterns
   □ Check for class imbalance issues
   □ Note diagonal vs off-diagonal patterns

3. PER-CLASS PERFORMANCE ANALYSIS
   □ Calculate precision, recall, F1 for each class
   □ Identify worst-performing classes
   □ Check for classes with high precision but low recall (or vice versa)
   □ Look for classes with consistently poor performance
   □ Rank classes by performance metrics

4. MISCLASSIFICATION PATTERN ANALYSIS
   □ Identify most common error patterns
   □ Look for directional bias (A→B vs B→A)
   □ Check if errors are symmetric or asymmetric
   □ Identify if certain classes are frequently confused
   □ Note if errors follow semantic relationships

5. CONFIDENCE ANALYSIS
   □ Compare confidence distributions for correct vs incorrect predictions
   □ Check if model is overconfident on wrong predictions
   □ Identify if low confidence correlates with errors
   □ Look for calibration issues
   □ Determine if confidence can be used for rejection

6. FEATURE IMPORTANCE ANALYSIS
   □ Generate saliency maps for misclassified samples
   □ Identify what image regions the model focuses on
   □ Check if model attends to relevant features
   □ Look for attention to irrelevant background features
   □ Compare attention patterns between correct and incorrect predictions

7. DATA QUALITY INVESTIGATION
   □ Examine actual misclassified images
   □ Check for data quality issues (blurry, low contrast, etc.)
   □ Look for edge cases and outliers
   □ Identify if errors are due to data problems
   □ Check for annotation errors

8. MODEL BEHAVIOR ANALYSIS
   □ Analyze prediction probabilities
   □ Check for systematic bias in predictions
   □ Look for threshold effects
   □ Identify if model is too conservative or aggressive
   □ Check for class-specific biases

9. IMPROVEMENT PRIORITIZATION
   □ Rank issues by impact on overall performance
   □ Identify quick wins vs long-term improvements
   □ Prioritize classes with highest error rates
   □ Consider data augmentation needs
   □ Plan targeted interventions

10. ACTIONABLE INSIGHTS GENERATION
    □ Create specific improvement recommendations
    □ Identify data collection needs
    □ Plan model architecture changes
    □ Design targeted training strategies
    □ Set up monitoring for specific error patterns

INTERPRETATION GUIDELINES:
=========================

HIGH CONFUSION MATRIX VALUES OFF-DIAGONAL:
- Indicates systematic confusion between classes
- May suggest similar visual features
- Consider data augmentation or feature engineering

LOW CONFIDENCE ON CORRECT PREDICTIONS:
- Model lacks confidence even when right
- May need better training or calibration
- Consider ensemble methods

HIGH CONFIDENCE ON INCORRECT PREDICTIONS:
- Model is overconfident
- Serious calibration issue
- May need temperature scaling or better training

UNEVEN PER-CLASS PERFORMANCE:
- Some classes much worse than others
- May indicate data imbalance
- Consider class-weighted loss or data augmentation

SYSTEMATIC ERROR PATTERNS:
- Consistent A→B misclassifications
- May indicate feature confusion
- Consider feature engineering or architecture changes

FEATURE IMPORTANCE ISSUES:
- Model focuses on wrong image regions
- May need better architecture or training
- Consider attention mechanisms or better preprocessing

NEXT STEPS AFTER ANALYSIS:
==========================

1. DATA IMPROVEMENTS:
   - Collect more data for poorly performing classes
   - Improve data quality and annotation
   - Add data augmentation for underrepresented classes

2. MODEL IMPROVEMENTS:
   - Adjust architecture for specific issues
   - Implement class-weighted loss
   - Add attention mechanisms if needed
   - Try ensemble methods

3. TRAINING IMPROVEMENTS:
   - Adjust learning rate and schedule
   - Use better optimization techniques
   - Implement curriculum learning
   - Add regularization if overfitting

4. EVALUATION IMPROVEMENTS:
   - Set up monitoring for specific error patterns
   - Implement confidence-based rejection
   - Add human-in-the-loop validation
   - Create targeted test sets

This systematic approach ensures comprehensive understanding of model errors
and provides actionable insights for improvement.
"""

class ErrorAnalyzer:
    """
    Comprehensive error analysis for the AnimeOrNot classification model.
    
    This class provides tools to analyze model errors, understand failure modes,
    and identify areas for improvement.
    """
    
    def __init__(self, model, test_dataloader, device=None, class_names=None):
        """
        Initialize the error analyzer.
        
        Args:
            model: Trained PyTorch model
            test_dataloader: Test data loader
            device: Device to run inference on
            class_names: List of class names for better visualization
        """
        self.model = model
        self.test_dataloader = test_dataloader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names or [f"Class_{i}" for i in range(10)]
        
        # Store predictions and ground truth
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []
        self.misclassified_samples = []
        
    def collect_predictions(self):
        """Collect all predictions and ground truth labels."""
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Get predictions
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Store results
                self.all_predictions.extend(predictions.cpu().numpy())
                self.all_labels.extend(labels.cpu().numpy())
                self.all_probabilities.extend(probabilities.cpu().numpy())
                
                # Store misclassified samples
                for i, (pred, true_label) in enumerate(zip(predictions, labels)):
                    if pred != true_label:
                        self.misclassified_samples.append({
                            'batch_idx': batch_idx,
                            'sample_idx': i,
                            'predicted': pred.item(),
                            'true_label': true_label.item(),
                            'confidence': probabilities[i][pred].item(),
                            'true_confidence': probabilities[i][true_label].item()
                        })
        
        print(f"Collected predictions for {len(self.all_labels)} samples")
        print(f"Found {len(self.misclassified_samples)} misclassified samples")
        
    def confusion_matrix_analysis(self, save_path=None):
        """Generate and analyze confusion matrix."""
        cm = confusion_matrix(self.all_labels, self.all_predictions)
        
        # Calculate metrics
        accuracy = np.sum(np.array(self.all_predictions) == np.array(self.all_labels)) / len(self.all_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.all_labels, self.all_predictions, average='weighted'
        )
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def per_class_analysis(self):
        """Analyze performance for each class."""
        cm = confusion_matrix(self.all_labels, self.all_predictions)
        
        class_metrics = {}
        for i in range(len(self.class_names)):
            # True positives, false positives, false negatives
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = np.sum(cm) - tp - fp - fn
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            class_metrics[self.class_names[i]] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'support': np.sum(cm[i, :])
            }
        
        # Create summary DataFrame
        df = pd.DataFrame(class_metrics).T
        print("\nPer-Class Performance Analysis:")
        print(df.round(4))
        
        return class_metrics
    
    def misclassification_analysis(self, top_k=10):
        """Analyze the most common misclassification patterns."""
        if not self.misclassified_samples:
            print("No misclassified samples found.")
            return
        
        # Count misclassification patterns
        misclass_patterns = defaultdict(int)
        for sample in self.misclassified_samples:
            pattern = (sample['true_label'], sample['predicted'])
            misclass_patterns[pattern] += 1
        
        # Sort by frequency
        sorted_patterns = sorted(misclass_patterns.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop {top_k} Most Common Misclassification Patterns:")
        print("True Label -> Predicted Label | Count")
        print("-" * 50)
        
        for (true_label, pred_label), count in sorted_patterns[:top_k]:
            true_name = self.class_names[true_label]
            pred_name = self.class_names[pred_label]
            print(f"{true_name} -> {pred_name} | {count}")
        
        return sorted_patterns
    
    def confidence_analysis(self):
        """Analyze prediction confidence patterns."""
        if not self.misclassified_samples:
            print("No misclassified samples for confidence analysis.")
            return
        
        # Separate correct and incorrect predictions
        correct_confidences = []
        incorrect_confidences = []
        
        for i, (pred, true_label) in enumerate(zip(self.all_predictions, self.all_labels)):
            confidence = self.all_probabilities[i][pred]
            if pred == true_label:
                correct_confidences.append(confidence)
            else:
                incorrect_confidences.append(confidence)
        
        # Plot confidence distributions
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(correct_confidences, bins=20, alpha=0.7, label='Correct Predictions', color='green')
        plt.hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect Predictions', color='red')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot([correct_confidences, incorrect_confidences], 
                   labels=['Correct', 'Incorrect'])
        plt.ylabel('Confidence')
        plt.title('Confidence Comparison')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\nConfidence Statistics:")
        print(f"Correct predictions - Mean: {np.mean(correct_confidences):.4f}, Std: {np.std(correct_confidences):.4f}")
        print(f"Incorrect predictions - Mean: {np.mean(incorrect_confidences):.4f}, Std: {np.std(incorrect_confidences):.4f}")
        
        return {
            'correct_confidences': correct_confidences,
            'incorrect_confidences': incorrect_confidences
        }
    
    def feature_importance_analysis(self, sample_indices=None, num_samples=10):
        """
        Analyze feature importance using gradient-based methods.
        This helps understand which parts of the image the model focuses on.
        """
        if sample_indices is None:
            # Use random misclassified samples
            if len(self.misclassified_samples) > 0:
                indices = np.random.choice(len(self.misclassified_samples), 
                                         min(num_samples, len(self.misclassified_samples)), 
                                         replace=False)
                sample_indices = [self.misclassified_samples[i]['batch_idx'] for i in indices]
            else:
                print("No misclassified samples available for feature importance analysis.")
                return
        
        self.model.eval()
        
        # Enable gradient computation
        self.model.zero_grad()
        
        for batch_idx, (images, labels) in enumerate(self.test_dataloader):
            if batch_idx in sample_indices:
                images = images.to(self.device)
                labels = labels.to(self.device)
                images.requires_grad_(True)
                
                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                # Calculate gradients for misclassified samples
                for i, (pred, true_label) in enumerate(zip(predictions, labels)):
                    if pred != true_label:
                        # Backward pass
                        outputs[i, pred].backward()
                        
                        # Get gradients
                        gradients = images.grad[i]
                        
                        # Create saliency map
                        saliency_map = torch.abs(gradients).mean(dim=0)
                        
                        # Visualize
                        self._visualize_saliency(images[i], saliency_map, 
                                               true_label.item(), pred.item())
                
                break
    
    def _visualize_saliency(self, image, saliency_map, true_label, pred_label):
        """Visualize saliency map for a single image."""
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        img_np = image.cpu().permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        plt.imshow(img_np)
        plt.title(f'Original Image\nTrue: {self.class_names[true_label]}\nPred: {self.class_names[pred_label]}')
        plt.axis('off')
        
        # Saliency map
        plt.subplot(1, 3, 2)
        saliency_np = saliency_map.cpu().numpy()
        plt.imshow(saliency_np, cmap='hot')
        plt.title('Saliency Map')
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 3, 3)
        plt.imshow(img_np)
        plt.imshow(saliency_np, cmap='hot', alpha=0.6)
        plt.title('Saliency Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def generate_error_report(self, save_path="error_analysis_report.json"):
        """Generate a comprehensive error analysis report."""
        print("Generating comprehensive error analysis report...")
        
        # Collect all analyses
        report = {
            'summary': {
                'total_samples': len(self.all_labels),
                'misclassified_samples': len(self.misclassified_samples),
                'accuracy': np.sum(np.array(self.all_predictions) == np.array(self.all_labels)) / len(self.all_labels)
            },
            'confusion_matrix_analysis': self.confusion_matrix_analysis(),
            'per_class_analysis': self.per_class_analysis(),
            'misclassification_patterns': self.misclassification_analysis(),
            'confidence_analysis': self.confidence_analysis()
        }
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Error analysis report saved to {save_path}")
        return report
    
    def run_full_analysis(self, save_plots=True, output_dir="error_analysis"):
        """
        Run complete error analysis pipeline.
        
        Args:
            save_plots: Whether to save generated plots
            output_dir: Directory to save analysis outputs
        """
        print("Starting comprehensive error analysis...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Collect predictions
        print("1. Collecting predictions...")
        self.collect_predictions()
        
        # Step 2: Confusion matrix analysis
        print("2. Analyzing confusion matrix...")
        cm_path = os.path.join(output_dir, "confusion_matrix.png") if save_plots else None
        cm_results = self.confusion_matrix_analysis(cm_path)
        
        # Step 3: Per-class analysis
        print("3. Analyzing per-class performance...")
        class_metrics = self.per_class_analysis()
        
        # Step 4: Misclassification patterns
        print("4. Analyzing misclassification patterns...")
        misclass_patterns = self.misclassification_analysis()
        
        # Step 5: Confidence analysis
        print("5. Analyzing prediction confidence...")
        conf_path = os.path.join(output_dir, "confidence_analysis.png") if save_plots else None
        confidence_results = self.confidence_analysis()
        
        # Step 6: Generate comprehensive report
        print("6. Generating final report...")
        report_path = os.path.join(output_dir, "error_analysis_report.json")
        report = self.generate_error_report(report_path)
        
        print(f"\nError analysis completed! Results saved to {output_dir}")
        print(f"Key findings:")
        print(f"- Overall accuracy: {report['summary']['accuracy']:.4f}")
        print(f"- Misclassified samples: {report['summary']['misclassified_samples']}")
        print(f"- F1 score: {report['confusion_matrix_analysis']['f1_score']:.4f}")
        
        return report


def quick_error_analysis(model, test_dataloader, class_names=None):
    """
    Quick error analysis function for immediate insights.
    
    Args:
        model: Trained PyTorch model
        test_dataloader: Test data loader
        class_names: List of class names
    
    Returns:
        Dictionary with key error analysis metrics
    """
    analyzer = ErrorAnalyzer(model, test_dataloader, class_names=class_names)
    analyzer.collect_predictions()
    
    # Quick metrics
    accuracy = np.sum(np.array(analyzer.all_predictions) == np.array(analyzer.all_labels)) / len(analyzer.all_labels)
    
    # Most common errors
    misclass_patterns = analyzer.misclassification_analysis(top_k=5)
    
    # Per-class performance
    class_metrics = analyzer.per_class_analysis()
    
    return {
        'accuracy': accuracy,
        'total_samples': len(analyzer.all_labels),
        'misclassified_count': len(analyzer.misclassified_samples),
        'top_misclassifications': misclass_patterns[:5],
        'worst_performing_class': min(class_metrics.items(), key=lambda x: x[1]['f1_score'])[0]
    }


# Example usage
if __name__ == "__main__":
    # This would be used after training your model
    print("Error Analysis Module for AnimeOrNot Project")
    print("=" * 50)
    print("\nUsage:")
    print("1. Initialize analyzer: analyzer = ErrorAnalyzer(model, test_loader)")
    print("2. Run full analysis: analyzer.run_full_analysis()")
    print("3. Quick analysis: quick_error_analysis(model, test_loader)")
    print("\nThis module provides:")
    print("- Confusion matrix analysis")
    print("- Per-class performance metrics")
    print("- Misclassification pattern analysis")
    print("- Confidence analysis")
    print("- Feature importance visualization")
    print("- Comprehensive error reports") 