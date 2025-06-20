import pandas as pd
from sklearn.metrics import classification_report
from collections import Counter

def print_detailed_summary(y_true, y_pred, dataset_name="Test"):
    """
    Print a human-readable summary of classification results
    """
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    
    print(f"\n" + "="*60)
    print(f"📊 DETAILED CLASSIFICATION SUMMARY - {dataset_name.upper()} SET")
    print("="*60)
    
    # Overall metrics
    accuracy = report_dict['accuracy']
    macro_f1 = report_dict['macro avg']['f1-score']
    weighted_f1 = report_dict['weighted avg']['f1-score']
    
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   • Overall Accuracy: {accuracy:.1%}")
    print(f"   • Macro F1-Score: {macro_f1:.3f}")
    print(f"   • Weighted F1-Score: {weighted_f1:.3f}")
    
    # Per-class analysis
    class_metrics = {k: v for k, v in report_dict.items() 
                    if k not in ['accuracy', 'macro avg', 'weighted avg']}
    
    df_metrics = pd.DataFrame(class_metrics).T
    
    # Performance categories
    excellent = df_metrics[df_metrics['f1-score'] >= 0.9]
    good = df_metrics[(df_metrics['f1-score'] >= 0.8) & (df_metrics['f1-score'] < 0.9)]
    fair = df_metrics[(df_metrics['f1-score'] >= 0.7) & (df_metrics['f1-score'] < 0.8)]
    poor = df_metrics[df_metrics['f1-score'] < 0.7]
    
    print(f"\n📈 PERFORMANCE BREAKDOWN:")
    print(f"   • Excellent (F1 ≥ 0.9): {len(excellent)} classes ({len(excellent)/len(df_metrics)*100:.1f}%)")
    print(f"   • Good (0.8 ≤ F1 < 0.9): {len(good)} classes ({len(good)/len(df_metrics)*100:.1f}%)")
    print(f"   • Fair (0.7 ≤ F1 < 0.8): {len(fair)} classes ({len(fair)/len(df_metrics)*100:.1f}%)")
    print(f"   • Poor (F1 < 0.7): {len(poor)} classes ({len(poor)/len(df_metrics)*100:.1f}%)")
    
    # Best performing classes
    print(f"\n🏆 TOP 5 BEST PERFORMING CLASSES:")
    top_5 = df_metrics.nlargest(5, 'f1-score')
    for i, (intent, metrics) in enumerate(top_5.iterrows(), 1):
        print(f"   {i}. {intent}: F1={metrics['f1-score']:.3f} (P={metrics['precision']:.3f}, R={metrics['recall']:.3f})")
    
    # Worst performing classes
    print(f"\n⚠️  BOTTOM 5 CLASSES NEEDING ATTENTION:")
    bottom_5 = df_metrics.nsmallest(5, 'f1-score')
    for i, (intent, metrics) in enumerate(bottom_5.iterrows(), 1):
        print(f"   {i}. {intent}: F1={metrics['f1-score']:.3f} (P={metrics['precision']:.3f}, R={metrics['recall']:.3f})")
    
    # Precision vs Recall analysis
    high_precision_low_recall = df_metrics[(df_metrics['precision'] >= 0.9) & (df_metrics['recall'] < 0.7)]
    high_recall_low_precision = df_metrics[(df_metrics['recall'] >= 0.9) & (df_metrics['precision'] < 0.7)]
    
    if len(high_precision_low_recall) > 0:
        print(f"\n🎯 HIGH PRECISION, LOW RECALL (Conservative predictions):")
        for intent in high_precision_low_recall.index[:3]:
            metrics = high_precision_low_recall.loc[intent]
            print(f"   • {intent}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
    
    if len(high_recall_low_precision) > 0:
        print(f"\n🔄 HIGH RECALL, LOW PRECISION (Liberal predictions):")
        for intent in high_recall_low_precision.index[:3]:
            metrics = high_recall_low_precision.loc[intent]
            print(f"   • {intent}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
    
    print(f"\n" + "="*60)
    
    return df_metrics
    ax3.set_xlabel('Precision')
    ax3.set_ylabel('Recall')
    ax3.set_title('Precision vs Recall (colored by F1-score)')
    ax3.grid(alpha=0.3)
    
    # Add diagonal line for reference
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('F1-Score')
    
    # 4. Class distribution and performance
    ax4 = axes[1, 1]
    
    # Count actual vs predicted distributions
    actual_counts = Counter(y_true)
    predicted_counts = Counter(y_pred)
    
    # Get top 15 most common classes
    top_15_classes = [class_name for class_name, _ in Counter(y_true).most_common(15)]
    
    actual_top15 = [actual_counts[cls] for cls in top_15_classes]
    predicted_top15 = [predicted_counts.get(cls, 0) for cls in top_15_classes]
    
    x_pos = np.arange(len(top_15_classes))
    width = 0.35
    
    ax4.bar(x_pos - width/2, actual_top15, width, label='Actual', alpha=0.7, color='blue')
    ax4.bar(x_pos + width/2, predicted_top15, width, label='Predicted', alpha=0.7, color='orange')
    
    ax4.set_xlabel('Intent Classes')
    ax4.set_ylabel('Count')
    ax4.set_title('Top 15 Classes: Actual vs Predicted Distribution')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(top_15_classes, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df_metrics

def print_detailed_summary(y_true, y_pred, dataset_name="Test"):
    """
    Print a human-readable summary of classification results
    """
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    
    print(f"\n" + "="*60)
    print(f"📊 DETAILED CLASSIFICATION SUMMARY - {dataset_name.upper()} SET")
    print("="*60)
    
    # Overall metrics
    accuracy = report_dict['accuracy']
    macro_f1 = report_dict['macro avg']['f1-score']
    weighted_f1 = report_dict['weighted avg']['f1-score']
    
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   • Overall Accuracy: {accuracy:.1%}")
    print(f"   • Macro F1-Score: {macro_f1:.3f}")
    print(f"   • Weighted F1-Score: {weighted_f1:.3f}")
    
    # Per-class analysis
    class_metrics = {k: v for k, v in report_dict.items() 
                    if k not in ['accuracy', 'macro avg', 'weighted avg']}
    
    df_metrics = pd.DataFrame(class_metrics).T
    
    # Performance categories
    excellent = df_metrics[df_metrics['f1-score'] >= 0.9]
    good = df_metrics[(df_metrics['f1-score'] >= 0.8) & (df_metrics['f1-score'] < 0.9)]
    fair = df_metrics[(df_metrics['f1-score'] >= 0.7) & (df_metrics['f1-score'] < 0.8)]
    poor = df_metrics[df_metrics['f1-score'] < 0.7]
    
    print(f"\n📈 PERFORMANCE BREAKDOWN:")
    print(f"   • Excellent (F1 ≥ 0.9): {len(excellent)} classes ({len(excellent)/len(df_metrics)*100:.1f}%)")
    print(f"   • Good (0.8 ≤ F1 < 0.9): {len(good)} classes ({len(good)/len(df_metrics)*100:.1f}%)")
    print(f"   • Fair (0.7 ≤ F1 < 0.8): {len(fair)} classes ({len(fair)/len(df_metrics)*100:.1f}%)")
    print(f"   • Poor (F1 < 0.7): {len(poor)} classes ({len(poor)/len(df_metrics)*100:.1f}%)")
    
    # Best performing classes
    print(f"\n🏆 TOP 5 BEST PERFORMING CLASSES:")
    top_5 = df_metrics.nlargest(5, 'f1-score')
    for i, (intent, metrics) in enumerate(top_5.iterrows(), 1):
        print(f"   {i}. {intent}: F1={metrics['f1-score']:.3f} (P={metrics['precision']:.3f}, R={metrics['recall']:.3f})")
    
    # Worst performing classes
    print(f"\n⚠️  BOTTOM 5 CLASSES NEEDING ATTENTION:")
    bottom_5 = df_metrics.nsmallest(5, 'f1-score')
    for i, (intent, metrics) in enumerate(bottom_5.iterrows(), 1):
        print(f"   {i}. {intent}: F1={metrics['f1-score']:.3f} (P={metrics['precision']:.3f}, R={metrics['recall']:.3f})")
    
    # Precision vs Recall analysis
    high_precision_low_recall = df_metrics[(df_metrics['precision'] >= 0.9) & (df_metrics['recall'] < 0.7)]
    high_recall_low_precision = df_metrics[(df_metrics['recall'] >= 0.9) & (df_metrics['precision'] < 0.7)]
    
    if len(high_precision_low_recall) > 0:
        print(f"\n🎯 HIGH PRECISION, LOW RECALL (Conservative predictions):")
        for intent in high_precision_low_recall.index[:3]:
            metrics = high_precision_low_recall.loc[intent]
            print(f"   • {intent}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
    
    if len(high_recall_low_precision) > 0:
        print(f"\n🔄 HIGH RECALL, LOW PRECISION (Liberal predictions):")
        for intent in high_recall_low_precision.index[:3]:
            metrics = high_recall_low_precision.loc[intent]
            print(f"   • {intent}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
    
    print(f"\n" + "="*60)
    
    return df_metrics

def show_confusion_matrix_sample(y_true, y_pred, sample_classes=10):
    """
    Show a sample confusion matrix for most common classes
    """
    # Get most common classes
    most_common = Counter(y_true).most_common(sample_classes)
    common_classes = [cls for cls, _ in most_common]
    
    # Filter data for these classes only
    mask = pd.Series(y_true).isin(common_classes)
    y_true_filtered = pd.Series(y_true)[mask]
    y_pred_filtered = pd.Series(y_pred)[mask]
    
    # Create confusion matrix
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=common_classes)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=common_classes, yticklabels=common_classes)
    plt.title(f'Confusion Matrix - Top {sample_classes} Most Common Classes')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
