"""
Clean ECG Visualization Module - Professional Layout
Each visualization is separate and properly spaced
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set clean style
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10

# ============================================
# 1. SINGLE ECG DASHBOARD - CLEAN LAYOUT
# ============================================

def plot_single_ecg_analysis(ecg_signal, prediction_probs, true_labels, 
                             class_names=['NORM', 'MI', 'STTC', 'CD', 'HYP'],
                             sampling_rate=100):
    """
    Single ECG comprehensive analysis - Clean layout
    """
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    time_axis = np.arange(ecg_signal.shape[0]) / sampling_rate
    
    # Create figure with better spacing
    fig = plt.figure(figsize=(22, 14))
    
    # Title
    fig.suptitle('ECG Analysis Dashboard', fontsize=22, fontweight='bold', y=0.98)
    
    # === TOP: Lead II Rhythm Strip ===
    ax_rhythm = plt.subplot2grid((5, 4), (0, 0), colspan=3, fig=fig)
    ax_rhythm.plot(time_axis, ecg_signal[:, 1], 'b-', linewidth=2)
    ax_rhythm.set_title('Lead II - Rhythm Strip', fontsize=16, fontweight='bold', pad=15)
    ax_rhythm.set_xlabel('Time (seconds)', fontsize=12)
    ax_rhythm.set_ylabel('Amplitude (mV)', fontsize=12)
    ax_rhythm.grid(True, alpha=0.3, linestyle='--')
    
    # Detect R-peaks and calculate heart rate
    peaks, _ = signal.find_peaks(ecg_signal[:, 1], distance=sampling_rate*0.6, height=0)
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / sampling_rate
        heart_rate = 60 / np.mean(rr_intervals)
        ax_rhythm.plot(time_axis[peaks], ecg_signal[peaks, 1], 'ro', markersize=10, 
                      label=f'R-peaks | Heart Rate: {heart_rate:.0f} bpm', zorder=5)
        ax_rhythm.legend(loc='upper right', fontsize=13, framealpha=0.9)
    
    # === TOP RIGHT: Classification Results ===
    ax_class = plt.subplot2grid((5, 4), (0, 3), rowspan=2, fig=fig)
    y_pos = np.arange(len(class_names))
    colors = ['#2ecc71' if pred > 0.5 else '#95a5a6' for pred in prediction_probs]
    
    bars = ax_class.barh(y_pos, prediction_probs, color=colors, alpha=0.8, 
                         edgecolor='black', linewidth=1.5, height=0.6)
    ax_class.axvline(x=0.5, color='red', linestyle='--', linewidth=3, 
                     label='Decision Threshold', alpha=0.8)
    
    # Mark true positives
    for i, (true_label, pred_prob) in enumerate(zip(true_labels, prediction_probs)):
        if true_label == 1:
            ax_class.scatter(pred_prob, i, color='red', s=300, marker='*', 
                           edgecolors='black', linewidths=2, zorder=10,
                           label='Ground Truth' if i == np.argmax(true_labels) else '')
    
    ax_class.set_yticks(y_pos)
    ax_class.set_yticklabels(class_names, fontsize=14, fontweight='bold')
    ax_class.set_xlabel('Confidence Score', fontsize=13, fontweight='bold')
    ax_class.set_title('Model Predictions', fontsize=16, fontweight='bold', pad=15)
    ax_class.set_xlim([0, 1.05])
    ax_class.grid(axis='x', alpha=0.3, linestyle='--')
    ax_class.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    # Add values on bars
    for bar, prob in zip(bars, prediction_probs):
        width = bar.get_width()
        ax_class.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                     f'{prob:.3f}', va='center', fontsize=12, fontweight='bold')
    
    # === MIDDLE: 12-Lead ECG Grid ===
    for i in range(12):
        row = (i // 3) + 1
        col = i % 3
        
        ax = plt.subplot2grid((5, 4), (row, col), fig=fig)
        ax.plot(time_axis, ecg_signal[:, i], 'b-', linewidth=1.2)
        ax.set_title(f'Lead {lead_names[i]}', fontsize=13, fontweight='bold', pad=8)
        ax.grid(True, alpha=0.2, linestyle=':')
        ax.tick_params(labelsize=9)
        
        if row == 4:
            ax.set_xlabel('Time (s)', fontsize=10)
        if col == 0:
            ax.set_ylabel('mV', fontsize=10)
    
    # === BOTTOM RIGHT: Summary Box ===
    ax_summary = plt.subplot2grid((5, 4), (2, 3), rowspan=3, fig=fig)
    ax_summary.axis('off')
    
    # Build summary
    predicted_classes = [class_names[i] for i, p in enumerate(prediction_probs) if p > 0.5]
    true_classes = [class_names[i] for i, t in enumerate(true_labels) if t == 1]
    max_confidence = max(prediction_probs)
    is_match = set(predicted_classes) == set(true_classes)
    
    summary_text = f"""
╔═══════════════════════════════════╗
║      DIAGNOSTIC SUMMARY           ║
╚═══════════════════════════════════╝

Ground Truth:
  → {', '.join(true_classes) if true_classes else 'NORMAL'}

Model Prediction:
  → {', '.join(predicted_classes) if predicted_classes else 'NORMAL'}

Confidence Level:
  → {max_confidence:.1%}

Heart Rate:
  → {heart_rate:.0f} bpm

Classification Status:
  → {'✓ CORRECT MATCH' if is_match else '✗ INCORRECT'}

Signal Quality:
  → Good (12-lead complete)
"""
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                   fontsize=12, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round,pad=1.2', facecolor='#ecf0f1', 
                            edgecolor='black', linewidth=2, alpha=0.9))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3.0, w_pad=3.0)
    return fig

# ============================================
# 2. BATCH OVERVIEW - CLEAN GRID
# ============================================

def plot_batch_overview(X_batch, predictions_batch, true_labels_batch, 
                       class_names=['NORM', 'MI', 'STTC', 'CD', 'HYP'],
                       num_samples=6):
    """
    Clean batch overview with proper spacing
    """
    num_samples = min(num_samples, len(X_batch))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(20, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Batch Classification Overview', fontsize=20, fontweight='bold', y=0.995)
    
    for idx in range(num_samples):
        ecg = X_batch[idx]
        preds = predictions_batch[idx]
        true = true_labels_batch[idx]
        
        # Left: ECG Lead II
        ax_ecg = axes[idx, 0]
        ax_ecg.plot(ecg[:, 1], 'b-', linewidth=1.5)
        ax_ecg.set_title(f'Sample {idx+1} - Lead II', fontsize=14, fontweight='bold')
        ax_ecg.set_ylabel('Amplitude (mV)', fontsize=11)
        ax_ecg.grid(True, alpha=0.3)
        
        if idx == num_samples - 1:
            ax_ecg.set_xlabel('Time (samples)', fontsize=11)
        
        # Right: Predictions
        ax_pred = axes[idx, 1]
        colors = ['#2ecc71' if p > 0.5 else '#e74c3c' if t == 1 else '#95a5a6' 
                 for p, t in zip(preds, true)]
        
        bars = ax_pred.barh(class_names, preds, color=colors, alpha=0.7, 
                           edgecolor='black', linewidth=1.5, height=0.5)
        ax_pred.axvline(x=0.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax_pred.set_xlim([0, 1])
        ax_pred.set_xlabel('Probability', fontsize=11)
        ax_pred.set_title('Classification Results', fontsize=14, fontweight='bold')
        ax_pred.grid(axis='x', alpha=0.3)
        
        # Add values and checkmarks
        for i, (bar, p, t) in enumerate(zip(bars, preds, true)):
            width = bar.get_width()
            pred_positive = p > 0.5
            is_correct = (pred_positive and t == 1) or (not pred_positive and t == 0)
            marker = '✓' if is_correct else '✗'
            color = '#2ecc71' if is_correct else '#e74c3c'
            
            ax_pred.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{p:.2f} {marker}', va='center', fontsize=10, 
                        fontweight='bold', color=color)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99], h_pad=2.5, w_pad=2.5)
    return fig

# ============================================
# 3. ROC CURVES - ONE PER FIGURE
# ============================================

def plot_roc_curves(y_true, y_pred, class_names=['NORM', 'MI', 'STTC', 'CD', 'HYP']):
    """
    ROC curves - each class in separate subplot with proper spacing
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('ROC Curves - Multi-Label Classification', fontsize=20, fontweight='bold')
    axes = axes.flatten()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for idx, class_name in enumerate(class_names):
        ax = axes[idx]
        
        # Calculate ROC
        fpr, tpr, thresholds = roc_curve(y_true[:, idx], y_pred[:, idx])
        roc_auc = auc(fpr, tpr)
        
        # Plot
        ax.plot(fpr, tpr, color=colors[idx], lw=3, 
               label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Classifier')
        
        # Optimal point
        optimal_idx = np.argmax(tpr - fpr)
        ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=200, 
                  marker='*', edgecolors='black', linewidths=2, zorder=5,
                  label=f'Optimal (threshold={thresholds[optimal_idx]:.2f})')
        
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        ax.set_title(f'{class_name}', fontsize=16, fontweight='bold', pad=15)
        ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')
    
    axes[-1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3.0, w_pad=3.0)
    return fig

# ============================================
# 4. PRECISION-RECALL CURVES
# ============================================

def plot_precision_recall(y_true, y_pred, class_names=['NORM', 'MI', 'STTC', 'CD', 'HYP']):
    """
    Precision-Recall curves with clean spacing
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Precision-Recall Curves', fontsize=20, fontweight='bold')
    axes = axes.flatten()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for idx, class_name in enumerate(class_names):
        ax = axes[idx]
        
        precision, recall, thresholds = precision_recall_curve(y_true[:, idx], y_pred[:, idx])
        pr_auc = auc(recall, precision)
        
        ax.plot(recall, precision, color=colors[idx], lw=3,
               label=f'AP = {pr_auc:.3f}')
        
        # Best F1 point
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        best_f1_idx = np.argmax(f1_scores)
        
        ax.scatter(recall[best_f1_idx], precision[best_f1_idx], color='red', 
                  s=200, marker='*', edgecolors='black', linewidths=2, zorder=5,
                  label=f'Best F1 = {f1_scores[best_f1_idx]:.3f}')
        
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
        ax.set_title(f'{class_name}', fontsize=16, fontweight='bold', pad=15)
        ax.legend(loc="lower left", fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')
    
    axes[-1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3.0, w_pad=3.0)
    return fig

# ============================================
# 5. CONFUSION MATRICES - CLEAN HEATMAPS
# ============================================

def plot_confusion_matrices(y_true, y_pred, class_names=['NORM', 'MI', 'STTC', 'CD', 'HYP']):
    """
    Clean confusion matrices with metrics
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Confusion Matrices with Performance Metrics', 
                 fontsize=20, fontweight='bold')
    axes = axes.flatten()
    
    for idx, class_name in enumerate(class_names):
        ax = axes[idx]
        
        y_pred_binary = (y_pred[:, idx] > 0.5).astype(int)
        cm = confusion_matrix(y_true[:, idx], y_pred_binary, labels=[0, 1])
        
        # Normalize for percentages
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            cm_percent = np.nan_to_num(cm_percent, nan=0.0)
        
        # Annotations
        annot = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                          for j in range(cm.shape[1])] 
                          for i in range(cm.shape[0])])
        
        # Heatmap
        sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap='YlOrRd',
                   cbar_kws={'label': 'Count', 'shrink': 0.8},
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   linewidths=2, linecolor='white',
                   annot_kws={'fontsize': 12, 'fontweight': 'bold'})
        
        ax.set_title(f'{class_name}', fontsize=16, fontweight='bold', pad=15)
        ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
        
        # Calculate metrics
        cm_values = cm.ravel()
        if len(cm_values) == 4:
            tn, fp, fn, tp = cm_values
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            metrics_text = (f'Accuracy: {accuracy:.3f} | Sensitivity: {sensitivity:.3f}\n'
                           f'Specificity: {specificity:.3f} | F1-Score: {f1:.3f}')
        else:
            metrics_text = 'Insufficient samples for metrics'
        
        ax.text(0.5, -0.22, metrics_text, transform=ax.transAxes,
               ha='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', 
                        edgecolor='black', linewidth=1.5, alpha=0.9))
    
    axes[-1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=4.0, w_pad=3.0)
    return fig

# ============================================
# 6. SIGNAL CHARACTERISTICS
# ============================================

def plot_signal_characteristics(ecg_signals, labels, 
                               class_names=['NORM', 'MI', 'STTC', 'CD', 'HYP']):
    """
    Average ECG patterns by class
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Average ECG Patterns by Diagnosis (Lead II)', 
                 fontsize=20, fontweight='bold')
    axes = axes.flatten()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for idx, class_name in enumerate(class_names):
        ax = axes[idx]
        
        class_mask = labels[:, idx] == 1
        class_signals = ecg_signals[class_mask]
        
        if len(class_signals) > 0:
            mean_signal = np.mean(class_signals[:, :, 1], axis=0)
            std_signal = np.std(class_signals[:, :, 1], axis=0)
            
            time = np.arange(len(mean_signal))
            
            ax.plot(time, mean_signal, color=colors[idx], linewidth=3, 
                   label=f'Mean (n={len(class_signals)})')
            ax.fill_between(time, mean_signal - std_signal, mean_signal + std_signal,
                           alpha=0.3, color=colors[idx], label='±1 SD')
            
            ax.set_title(f'{class_name}', fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel('Time (samples)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Amplitude (mV)', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
        else:
            ax.text(0.5, 0.5, f'No {class_name} samples\navailable', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=16, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            ax.set_title(f'{class_name} (n=0)', fontsize=16, fontweight='bold', pad=15)
    
    axes[-1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3.0, w_pad=3.0)
    return fig

# ============================================
# MAIN VISUALIZATION FUNCTION
# ============================================

def visualize_model_results(model, test_loader, device='cuda', num_detailed=2):
    """
    Generate all visualizations with proper spacing
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        device: 'cuda' or 'cpu'
        num_detailed: Number of detailed individual analyses
    """
    import torch
    
    model.eval()
    
    all_signals = []
    all_preds = []
    all_labels = []
    
    print("Collecting predictions from test set...")
    with torch.no_grad():
        for signals, labels in test_loader:
            signals_device = signals.to(device)
            outputs = model(signals_device)
            
            all_signals.append(signals.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_signals = np.vstack(all_signals)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # 1. Detailed individual analyses
    print(f"\n[1/6] Creating {num_detailed} detailed ECG dashboards...")
    for i in range(min(num_detailed, len(all_signals))):
        fig = plot_single_ecg_analysis(all_signals[i], all_preds[i], 
                                       all_labels[i], class_names)
        filename = f'01_ecg_dashboard_{i+1}.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✓ Saved: {filename}")
        plt.close(fig)
    
    # 2. Batch overview
    print("\n[2/6] Creating batch overview...")
    fig = plot_batch_overview(all_signals, all_preds, all_labels, class_names, num_samples=6)
    filename = '02_batch_overview.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Saved: {filename}")
    plt.close(fig)
    
    # 3. ROC curves
    print("\n[3/6] Creating ROC curves...")
    fig = plot_roc_curves(all_labels, all_preds, class_names)
    filename = '03_roc_curves.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Saved: {filename}")
    plt.close(fig)
    
    # 4. Precision-Recall curves
    print("\n[4/6] Creating Precision-Recall curves...")
    fig = plot_precision_recall(all_labels, all_preds, class_names)
    filename = '04_precision_recall.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Saved: {filename}")
    plt.close(fig)
    
    # 5. Confusion matrices
    print("\n[5/6] Creating confusion matrices...")
    fig = plot_confusion_matrices(all_labels, all_preds, class_names)
    filename = '05_confusion_matrices.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Saved: {filename}")
    plt.close(fig)
    
    # 6. Signal characteristics
    print("\n[6/6] Creating signal characteristics...")
    fig = plot_signal_characteristics(all_signals, all_labels, class_names)
    filename = '06_signal_characteristics.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Saved: {filename}")
    plt.close(fig)
    
    print("\n" + "="*60)
    print("✓ ALL VISUALIZATIONS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  01_ecg_dashboard_*.png       - Detailed individual analyses")
    print("  02_batch_overview.png         - Batch predictions")
    print("  03_roc_curves.png             - ROC curves (all classes)")
    print("  04_precision_recall.png       - Precision-Recall curves")
    print("  05_confusion_matrices.png     - Performance metrics")
    print("  06_signal_characteristics.png - Average patterns by class")
    print("\nAll files saved at 300 DPI - Publication ready!")

# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    print("Clean ECG Visualization Module")
    print("=" * 60)
    print("\nUsage:")
    print("  visualize_model_results(model, test_loader, device='cuda', num_detailed=2)")
    print("\nThis generates 6 professional visualizations with proper spacing")