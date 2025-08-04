#!/usr/bin/env python3
"""
Error Analysis Example for AnimeOrNot Project

This script demonstrates how to use the error analysis module
to understand model performance and identify improvement areas.
"""

import sys
import os
sys.path.append('src')

from src.data.load_data import load_image_dataset, dataloader_generator
from src.model.train_model import Trainer
from src.utils.error_analysis import ErrorAnalyzer, quick_error_analysis

def main():
    """Demonstrate error analysis for the AnimeOrNot project."""
    
    print("=" * 60)
    print("ERROR ANALYSIS FOR ANIMEORNOT PROJECT")
    print("=" * 60)
    
    # Step 1: Load your trained model and data
    print("\n1. Loading dataset and creating dataloaders...")
    
    # Load datasets
    train_data, test_data = load_image_dataset("/Users/tanzimfarhan/Desktop/Python/Dataset")
    
    # Create dataloaders
    train_loader = dataloader_generator(train_data, test_data, type="train", batch_size=32)
    test_loader = dataloader_generator(train_data, test_data, type="test", batch_size=32)
    
    # Step 2: Initialize trainer (assuming you have a trained model)
    print("\n2. Initializing trainer...")
    
    trainer = Trainer(
        num_classes=10,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        lr=0.001,
        device=None,
        output_shape=10,
        model_save_path="best_model.pth"
    )
    
    # Step 3: Quick Error Analysis (if you have a trained model)
    print("\n3. Running quick error analysis...")
    
    # Define class names (replace with your actual class names)
    class_names = [
        "Class_0", "Class_1", "Class_2", "Class_3", "Class_4",
        "Class_5", "Class_6", "Class_7", "Class_8", "Class_9"
    ]
    
    # Quick analysis (this will work even without a fully trained model)
    try:
        quick_results = quick_error_analysis(
            model=trainer.model,
            test_dataloader=test_loader,
            class_names=class_names
        )
        
        print("\nQuick Analysis Results:")
        print(f"Accuracy: {quick_results['accuracy']:.4f}")
        print(f"Total samples: {quick_results['total_samples']}")
        print(f"Misclassified: {quick_results['misclassified_count']}")
        print(f"Worst performing class: {quick_results['worst_performing_class']}")
        
    except Exception as e:
        print(f"Quick analysis failed (expected if model not trained): {e}")
    
    # Step 4: Comprehensive Error Analysis
    print("\n4. Setting up comprehensive error analyzer...")
    
    analyzer = ErrorAnalyzer(
        model=trainer.model,
        test_dataloader=test_loader,
        class_names=class_names
    )
    
    # Step 5: Run specific analyses
    print("\n5. Running specific analyses...")
    
    # Collect predictions
    print("   - Collecting predictions...")
    analyzer.collect_predictions()
    
    # Confusion matrix
    print("   - Generating confusion matrix...")
    cm_results = analyzer.confusion_matrix_analysis()
    print(f"   Overall accuracy: {cm_results['accuracy']:.4f}")
    print(f"   F1 score: {cm_results['f1_score']:.4f}")
    
    # Per-class analysis
    print("   - Analyzing per-class performance...")
    class_metrics = analyzer.per_class_analysis()
    
    # Misclassification patterns
    print("   - Analyzing misclassification patterns...")
    misclass_patterns = analyzer.misclassification_analysis(top_k=5)
    
    # Confidence analysis
    print("   - Analyzing prediction confidence...")
    confidence_results = analyzer.confidence_analysis()
    
    # Step 6: Generate comprehensive report
    print("\n6. Generating comprehensive report...")
    
    report = analyzer.generate_error_report("error_analysis_report.json")
    
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS COMPLETED!")
    print("=" * 60)
    
    print(f"\nKey Findings:")
    print(f"- Overall Accuracy: {report['summary']['accuracy']:.4f}")
    print(f"- Misclassified Samples: {report['summary']['misclassified_samples']}")
    print(f"- F1 Score: {report['confusion_matrix_analysis']['f1_score']:.4f}")
    print(f"- Precision: {report['confusion_matrix_analysis']['precision']:.4f}")
    print(f"- Recall: {report['confusion_matrix_analysis']['recall']:.4f}")
    
    print(f"\nFiles Generated:")
    print(f"- error_analysis_report.json: Comprehensive analysis report")
    print(f"- confusion_matrix.png: Visual confusion matrix")
    print(f"- confidence_analysis.png: Confidence distribution plots")
    
    print(f"\nNext Steps for Improvement:")
    print("1. Focus on classes with low F1 scores")
    print("2. Investigate most common misclassification patterns")
    print("3. Consider data augmentation for underrepresented classes")
    print("4. Analyze confidence patterns for model calibration")
    print("5. Use feature importance analysis to understand model decisions")

def demonstrate_usage():
    """Show how to use the error analysis module."""
    
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    
    print("\n1. Quick Analysis (for immediate insights):")
    print("""
    from src.utils.error_analysis import quick_error_analysis
    
    results = quick_error_analysis(
        model=your_trained_model,
        test_dataloader=test_loader,
        class_names=['class1', 'class2', ...]
    )
    print(f"Accuracy: {results['accuracy']:.4f}")
    """)
    
    print("\n2. Comprehensive Analysis (for detailed investigation):")
    print("""
    from src.utils.error_analysis import ErrorAnalyzer
    
    analyzer = ErrorAnalyzer(
        model=your_trained_model,
        test_dataloader=test_loader,
        class_names=['class1', 'class2', ...]
    )
    
    # Run full analysis pipeline
    report = analyzer.run_full_analysis(
        save_plots=True,
        output_dir="error_analysis_results"
    )
    """)
    
    print("\n3. Individual Analyses:")
    print("""
    # Confusion matrix
    cm_results = analyzer.confusion_matrix_analysis()
    
    # Per-class performance
    class_metrics = analyzer.per_class_analysis()
    
    # Misclassification patterns
    patterns = analyzer.misclassification_analysis(top_k=10)
    
    # Confidence analysis
    conf_results = analyzer.confidence_analysis()
    
    # Feature importance (for misclassified samples)
    analyzer.feature_importance_analysis(num_samples=5)
    """)

if __name__ == "__main__":
    # Run the demonstration
    main()
    
    # Show usage examples
    demonstrate_usage() 