#!/usr/bin/env python3
# github.com/rouze-d

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import warnings
import os
import sys
from io import StringIO
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Set style untuk matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class TOTO4DAnalyzer:
    def __init__(self, data_file=None, chunk_size=10000):
        """Initialize the TOTO 4D Analyzer"""
        self.data = None
        self.all_numbers_flat = []
        self.digit_data = []
        self.recent_data = None
        self.chunk_size = chunk_size
        self.color_palette = {
            'hot': '#FF6B6B',     # Merah untuk hot numbers
            'cold': '#4ECDC4',    # Hijau untuk cold numbers
            'neutral': '#FFD166', # Kuning untuk neutral
            'prediction': '#118AB2', # Biru untuk predictions
            'background': '#F7F7F7',
            'text': '#2D3047',
            'success': '#06D6A0',
            'warning': '#FFD166',
            'danger': '#EF476F'
        }
        
        # Custom colormap
        self.cmap_hot = LinearSegmentedColormap.from_list('hot_cmap', ['#FFD166', '#EF476F', '#FF6B6B'])
        self.cmap_cold = LinearSegmentedColormap.from_list('cold_cmap', ['#4ECDC4', '#118AB2', '#073B4C'])
        
        if data_file:
            self.load_data_large(data_file)
    
    def load_data_large(self, file_path):
        """Load historical data"""
        try:
            if not os.path.exists(file_path):
                print(f"❌ File '{file_path}' tidak ditemukan!")
                return False
            
            print(f"📂 Memuat data dari: {file_path}")
            
            # Read data
            chunks = []
            chunk_count = 0
            
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                chunk_count += 1
                chunks.append(chunk)
                
                if chunk_count % 10 == 0:
                    print(f"   Chunk {chunk_count}...")
                
                if chunk_count > 100:
                    break
            
            self.data = pd.concat(chunks, ignore_index=True)
            
            del chunks
            gc.collect()
            
            print(f"✅ Data berhasil dimuat: {len(self.data):,} rekaman")
            
            # Preprocessing
            success = self.preprocess_data_large()
            return success
            
        except Exception as e:
            print(f"❌ Error memuat data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def preprocess_data_large(self):
        """Preprocess data"""
        try:
            if 'Draw_Date' in self.data.columns:
                self.data['Draw_Date'] = pd.to_datetime(self.data['Draw_Date'], errors='coerce')
                self.data = self.data[self.data['Draw_Date'].notna()]
                self.data = self.data.sort_values('Draw_Date', ascending=True)
                
                print(f"📊 Total undian: {len(self.data):,}")
                
                self.recent_data = self.data.tail(1000).copy() if len(self.data) >= 1000 else self.data.copy()
            else:
                print("❌ Kolom 'Draw_Date' tidak ditemukan!")
                return False
            
            # Process numbers
            number_columns = [col for col in self.data.columns if col != 'Draw_Date']
            
            print(f"🔢 Memproses {len(number_columns)} kolom...")
            
            self.all_numbers_flat = []
            self.digit_data = []
            
            # Process in batches
            batch_size = 5000
            total_batches = (len(self.data) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(self.data))
                
                batch_data = self.data.iloc[start_idx:end_idx]
                
                for _, row in batch_data.iterrows():
                    for col in number_columns:
                        num_str = str(row[col]).strip()
                        if num_str and num_str != 'nan' and num_str != 'None':
                            num_str = num_str.zfill(4)[:4]
                            if len(num_str) == 4 and num_str.isdigit():
                                self.all_numbers_flat.append(num_str)
                                self.digit_data.append([int(d) for d in num_str])
                
                if batch_idx % 20 == 0:
                    print(f"   Progress: {(end_idx/len(self.data)*100):.1f}%")
            
            print(f"✅ Total angka diproses: {len(self.all_numbers_flat):,}")
            
            # Convert to numpy for efficiency
            self.all_numbers_flat = np.array(self.all_numbers_flat, dtype='U4')
            self.digit_data = np.array(self.digit_data, dtype=np.uint8)
            
            return True
            
        except Exception as e:
            print(f"❌ Error preprocessing data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ============================================
    # VISUALIZATION FUNCTIONS
    # ============================================
    
    def create_frequency_heatmap(self, top_n=20):
        """Create heatmap of most frequent numbers"""
        if len(self.all_numbers_flat) == 0:
            print("❌ Data belum diproses!")
            return None
        
        unique_values, counts = np.unique(self.all_numbers_flat, return_counts=True)
        
        # Get top N numbers
        top_indices = np.argsort(counts)[-top_n:][::-1]
        top_numbers = unique_values[top_indices]
        top_counts = counts[top_indices]
        
        # Create heatmap data
        heatmap_data = []
        for num, count in zip(top_numbers, top_counts):
            # Convert number to 2D array of digits
            digits = [int(d) for d in str(num).zfill(4)]
            heatmap_data.append(digits + [count])
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                  columns=['Digit1', 'Digit2', 'Digit3', 'Digit4', 'Frequency'])
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'📊 Heatmap Top {top_n} Angka Paling Sering Muncul', 
                    fontsize=16, fontweight='bold', color=self.color_palette['text'])
        
        # Plot 1: Bar chart frequencies
        ax1 = axes[0, 0]
        bars = ax1.barh(range(len(top_numbers)), top_counts, 
                       color=self.cmap_hot(np.linspace(0.2, 0.8, len(top_numbers))))
        ax1.set_yticks(range(len(top_numbers)))
        ax1.set_yticklabels(top_numbers)
        ax1.set_xlabel('Frekuensi', fontsize=12)
        ax1.set_title('Frekuensi Kemunculan', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, top_counts)):
            ax1.text(count + max(top_counts)*0.01, bar.get_y() + bar.get_height()/2, 
                    str(count), va='center', fontsize=10)
        
        # Plot 2: Heatmap of digits
        ax2 = axes[0, 1]
        digit_matrix = heatmap_df[['Digit1', 'Digit2', 'Digit3', 'Digit4']].values
        im = ax2.imshow(digit_matrix, cmap='viridis', aspect='auto')
        ax2.set_xticks(range(4))
        ax2.set_xticklabels(['Pos 1', 'Pos 2', 'Pos 3', 'Pos 4'])
        ax2.set_yticks(range(len(top_numbers)))
        ax2.set_yticklabels(top_numbers)
        ax2.set_title('Distribusi Digit per Posisi', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax2)
        
        # Plot 3: Frequency distribution
        ax3 = axes[1, 0]
        all_counts = counts[counts > 0]
        ax3.hist(all_counts, bins=30, alpha=0.7, 
                color=self.color_palette['prediction'], edgecolor='black')
        ax3.set_xlabel('Frekuensi', fontsize=12)
        ax3.set_ylabel('Jumlah Angka', fontsize=12)
        ax3.set_title('Distribusi Frekuensi Semua Angka', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Top numbers with color coding
        ax4 = axes[1, 1]
        colors = [self.color_palette['hot'] if count > np.median(top_counts) 
                 else self.color_palette['neutral'] for count in top_counts]
        ax4.bar(range(len(top_numbers)), top_counts, color=colors, edgecolor='black')
        ax4.set_xticks(range(len(top_numbers)))
        ax4.set_xticklabels(top_numbers, rotation=45, ha='right')
        ax4.set_xlabel('Angka', fontsize=12)
        ax4.set_ylabel('Frekuensi', fontsize=12)
        ax4.set_title('Top Numbers (Warna: 🔥 Hot | ⭐ Neutral)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('frequency_heatmap.png', dpi=150, bbox_inches='tight')
        print(f"✅ Heatmap disimpan sebagai 'frequency_heatmap.png'")
        plt.show()
        
        return fig
    
    def create_digit_distribution_chart(self):
        """Create visualization for digit distribution"""
        if len(self.digit_data) == 0:
            print("❌ Data belum diproses!")
            return None
        
        digit_data_np = np.array(self.digit_data)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribusi Digit Posisi 1', 'Distribusi Digit Posisi 2',
                           'Distribusi Digit Posisi 3', 'Distribusi Digit Posisi 4'),
            specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
                   [{'type': 'histogram'}, {'type': 'histogram'}]]
        )
        
        positions = ['Posisi 1', 'Posisi 2', 'Posisi 3', 'Posisi 4']
        colors = [self.color_palette['hot'], self.color_palette['prediction'],
                 self.color_palette['success'], self.color_palette['warning']]
        
        for i in range(4):
            row = i // 2 + 1
            col = i % 2 + 1
            
            digit_counts = np.bincount(digit_data_np[:, i], minlength=10)
            
            fig.add_trace(
                go.Bar(
                    x=list(range(10)),
                    y=digit_counts,
                    name=positions[i],
                    marker_color=colors[i],
                    text=digit_counts,
                    textposition='auto',
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text="Digit", row=row, col=col, tickvals=list(range(10)))
            fig.update_yaxes(title_text="Frekuensi", row=row, col=col)
        
        fig.update_layout(
            title_text="📊 Distribusi Digit per Posisi",
            showlegend=False,
            height=600,
            template='plotly_white',
            font=dict(size=12)
        )
        
        # Save as HTML
        fig.write_html("digit_distribution.html")
        print(f"✅ Chart distribusi digit disimpan sebagai 'digit_distribution.html'")
        
        # Also create matplotlib version
        fig_mat, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i in range(4):
            digit_counts = np.bincount(digit_data_np[:, i], minlength=10)
            axes[i].bar(range(10), digit_counts, color=colors[i], alpha=0.8, edgecolor='black')
            axes[i].set_xlabel('Digit', fontsize=11)
            axes[i].set_ylabel('Frekuensi', fontsize=11)
            axes[i].set_title(f'{positions[i]}', fontsize=13, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xticks(range(10))
            
            # Add value labels
            for j, count in enumerate(digit_counts):
                if count > 0:
                    axes[i].text(j, count, str(count), ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Distribusi Digit per Posisi', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('digit_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_hot_cold_chart(self, top_n=15):
        """Create hot vs cold numbers visualization"""
        if len(self.all_numbers_flat) == 0:
            print("❌ Data belum diproses!")
            return None
        
        unique_values, counts = np.unique(self.all_numbers_flat, return_counts=True)
        
        # Get hot and cold numbers
        hot_indices = np.argsort(counts)[-top_n:][::-1]
        cold_indices = np.argsort(counts)[:top_n]
        
        hot_numbers = unique_values[hot_indices]
        hot_counts = counts[hot_indices]
        cold_numbers = unique_values[cold_indices]
        cold_counts = counts[cold_indices]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Hot numbers chart
        hot_colors = self.cmap_hot(np.linspace(0.3, 0.9, len(hot_numbers)))
        bars1 = ax1.barh(range(len(hot_numbers)), hot_counts, color=hot_colors, edgecolor='black')
        ax1.set_yticks(range(len(hot_numbers)))
        ax1.set_yticklabels(hot_numbers)
        ax1.set_xlabel('Frekuensi', fontsize=12)
        ax1.set_title(f'🔥 TOP {top_n} HOT NUMBERS', fontsize=14, fontweight='bold', color=self.color_palette['hot'])
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add frequency labels
        for i, (bar, count) in enumerate(zip(bars1, hot_counts)):
            ax1.text(count + max(hot_counts)*0.01, bar.get_y() + bar.get_height()/2, 
                    str(count), va='center', fontsize=10, fontweight='bold')
        
        # Cold numbers chart
        cold_colors = self.cmap_cold(np.linspace(0.3, 0.9, len(cold_numbers)))
        bars2 = ax2.barh(range(len(cold_numbers)), cold_counts, color=cold_colors, edgecolor='black')
        ax2.set_yticks(range(len(cold_numbers)))
        ax2.set_yticklabels(cold_numbers)
        ax2.set_xlabel('Frekuensi', fontsize=12)
        ax2.set_title(f'❄️ TOP {top_n} COLD NUMBERS', fontsize=14, fontweight='bold', color=self.color_palette['cold'])
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add frequency labels
        for i, (bar, count) in enumerate(zip(bars2, cold_counts)):
            ax2.text(count + max(cold_counts)*0.01, bar.get_y() + bar.get_height()/2, 
                    str(count), va='center', fontsize=10, fontweight='bold')
        
        plt.suptitle('Hot vs Cold Numbers Analysis', fontsize=18, fontweight='bold', 
                    color=self.color_palette['text'], y=1.02)
        plt.tight_layout()
        plt.savefig('hot_cold_chart.png', dpi=150, bbox_inches='tight')
        print(f"✅ Hot vs Cold chart disimpan sebagai 'hot_cold_chart.png'")
        plt.show()
        
        # Create interactive plotly version
        fig_plotly = go.Figure()
        
        # Add hot numbers trace
        fig_plotly.add_trace(go.Bar(
            x=hot_counts,
            y=hot_numbers,
            name='Hot Numbers',
            orientation='h',
            marker_color=hot_colors,
            text=hot_counts,
            textposition='auto',
        ))
        
        # Add cold numbers trace
        fig_plotly.add_trace(go.Bar(
            x=cold_counts,
            y=cold_numbers,
            name='Cold Numbers',
            orientation='h',
            marker_color=cold_colors,
            text=cold_counts,
            textposition='auto',
        ))
        
        fig_plotly.update_layout(
            title=f'Hot vs Cold Numbers Comparison (Top {top_n} each)',
            xaxis_title='Frequency',
            yaxis_title='Numbers',
            barmode='group',
            height=600,
            template='plotly_white',
            showlegend=True
        )
        
        fig_plotly.write_html("hot_cold_interactive.html")
        print(f"✅ Chart interaktif disimpan sebagai 'hot_cold_interactive.html'")
        
        return fig
    
    def create_even_odd_pie_chart(self):
        """Create even/odd analysis pie charts"""
        if len(self.digit_data) == 0:
            print("❌ Data belum diproses!")
            return None
        
        digit_data_np = np.array(self.digit_data)
        
        # Calculate even/odd statistics
        even_counts = []
        odd_counts = []
        
        for i in range(4):
            even_count = np.sum(digit_data_np[:, i] % 2 == 0)
            odd_count = np.sum(digit_data_np[:, i] % 2 == 1)
            even_counts.append(even_count)
            odd_counts.append(odd_count)
        
        # Create pie charts
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        positions = ['Posisi 1', 'Posisi 2', 'Posisi 3', 'Posisi 4']
        colors_even = ['#4ECDC4', '#06D6A0', '#118AB2', '#073B4C']
        colors_odd = ['#FF6B6B', '#EF476F', '#FF9E6D', '#FFD166']
        
        for i in range(4):
            sizes = [even_counts[i], odd_counts[i]]
            labels = ['Genap', 'Ganjil']
            colors = [colors_even[i], colors_odd[i]]
            
            wedges, texts, autotexts = axes[i].pie(
                sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, shadow=True, explode=(0.05, 0.05),
                textprops={'fontsize': 11, 'fontweight': 'bold'}
            )
            
            axes[i].set_title(f'{positions[i]}\nGenap: {even_counts[i]:,} | Ganjil: {odd_counts[i]:,}', 
                            fontsize=13, fontweight='bold')
            
            # Style the percentage text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        plt.suptitle('Distribusi Genap vs Ganjil per Posisi', fontsize=18, 
                    fontweight='bold', color=self.color_palette['text'], y=1.02)
        plt.tight_layout()
        plt.savefig('even_odd_pie_chart.png', dpi=150, bbox_inches='tight')
        print(f"✅ Pie chart genap/ganjil disimpan sebagai 'even_odd_pie_chart.png'")
        plt.show()
        
        # Create summary bar chart
        fig2, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(4)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, even_counts, width, label='Genap', 
                      color=self.color_palette['cold'], edgecolor='black')
        bars2 = ax.bar(x + width/2, odd_counts, width, label='Ganjil', 
                      color=self.color_palette['hot'], edgecolor='black')
        
        ax.set_xlabel('Posisi Digit', fontsize=12)
        ax.set_ylabel('Jumlah', fontsize=12)
        ax.set_title('Perbandingan Genap vs Ganjil per Posisi', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(positions)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:,}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
        
        autolabel(bars1)
        autolabel(bars2)
        
        plt.tight_layout()
        plt.savefig('even_odd_bar_chart.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_predictions_visualization(self, predictions_data):
        """Create visualization for predictions"""
        if not predictions_data:
            print("❌ Tidak ada data predictions!")
            return None
        
        # Prepare data
        all_predictions = []
        prediction_sources = []
        
        for source, predictions in predictions_data.items():
            for pred in predictions:
                all_predictions.append(pred)
                prediction_sources.append(source)
        
        # Count predictions
        pred_counter = Counter(all_predictions)
        
        # Get top predictions
        top_predictions = pred_counter.most_common(20)
        top_numbers = [p[0] for p in top_predictions]
        top_counts = [p[1] for p in top_predictions]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Bar chart of top predictions
        colors = []
        for count in top_counts:
            if count >= 3:
                colors.append(self.color_palette['hot'])  # Very hot
            elif count == 2:
                colors.append(self.color_palette['warning'])  # Warm
            else:
                colors.append(self.color_palette['prediction'])  # Cool
        
        bars = ax1.barh(range(len(top_predictions)), top_counts, color=colors, edgecolor='black')
        ax1.set_yticks(range(len(top_predictions)))
        ax1.set_yticklabels(top_numbers)
        ax1.set_xlabel('Jumlah Analisis yang Merekomendasikan', fontsize=12)
        ax1.set_title('TOP 20 PREDICTIONS (berdasarkan konsensus)', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, top_counts)):
            ax1.text(count + 0.1, bar.get_y() + bar.get_height()/2, 
                    f"{count} analisis", va='center', fontsize=10, fontweight='bold')
        
        # Create heatmap of prediction sources
        if len(predictions_data) > 1:
            # Prepare data for heatmap
            sources = list(predictions_data.keys())
            pred_matrix = []
            
            for pred in top_numbers[:10]:  # Top 10 predictions
                row = []
                for source in sources:
                    row.append(1 if pred in predictions_data[source] else 0)
                pred_matrix.append(row)
            
            im = ax2.imshow(pred_matrix, cmap='YlOrRd', aspect='auto')
            ax2.set_xticks(range(len(sources)))
            ax2.set_xticklabels([s[:15] + '...' if len(s) > 15 else s for s in sources], 
                               rotation=45, ha='right')
            ax2.set_yticks(range(len(top_numbers[:10])))
            ax2.set_yticklabels(top_numbers[:10])
            ax2.set_title('Sumber Predictions (Heatmap)', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax2, label='Direkomendasikan (1=Ya, 0=Tidak)')
        
        plt.suptitle('ANALISIS PREDICTIONS - VISUALISASI', fontsize=18, fontweight='bold', 
                    color=self.color_palette['text'], y=1.02)
        plt.tight_layout()
        plt.savefig('predictions_visualization.png', dpi=150, bbox_inches='tight')
        print(f"✅ Visualisasi predictions disimpan sebagai 'predictions_visualization.png'")
        plt.show()
        
        # Create interactive sunburst chart
        fig_sunburst = go.Figure(go.Sunburst(
            labels=prediction_sources + all_predictions,
            parents=[''] * len(prediction_sources) + prediction_sources,
            values=[1] * len(prediction_sources) + [1] * len(all_predictions),
            branchvalues="total",
            marker=dict(
                colors=[self.cmap_hot(i/len(set(prediction_sources))) for i in range(len(set(prediction_sources)))] * len(prediction_sources) +
                       [self.cmap_cold(i/len(set(all_predictions))) for i in range(len(set(all_predictions)))] * len(all_predictions)
            )
        ))
        
        fig_sunburst.update_layout(
            title="Struktur Predictions per Analisis",
            height=700
        )
        
        fig_sunburst.write_html("predictions_sunburst.html")
        print(f"✅ Sunburst chart interaktif disimpan sebagai 'predictions_sunburst.html'")
        
        return fig
    
    def create_summary_dashboard(self):
        """Create comprehensive summary dashboard"""
        if len(self.all_numbers_flat) == 0:
            print("❌ Data belum diproses!")
            return None
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Frequency distribution (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        unique_values, counts = np.unique(self.all_numbers_flat, return_counts=True)
        top_indices = np.argsort(counts)[-20:][::-1]
        top_numbers = unique_values[top_indices]
        top_counts = counts[top_indices]
        
        colors1 = self.cmap_hot(np.linspace(0.2, 0.8, len(top_numbers)))
        ax1.barh(range(len(top_numbers)), top_counts, color=colors1, edgecolor='black')
        ax1.set_yticks(range(len(top_numbers)))
        ax1.set_yticklabels(top_numbers)
        ax1.set_xlabel('Frekuensi')
        ax1.set_title('TOP 20 ANGKA TERPOPULER', fontweight='bold', color=self.color_palette['hot'])
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Digit position distribution
        ax2 = fig.add_subplot(gs[0, 1])
        digit_data_np = np.array(self.digit_data)
        digit_counts = []
        for i in range(4):
            digit_counts.append(np.bincount(digit_data_np[:, i], minlength=10))
        
        x = np.arange(10)
        width = 0.2
        positions = ['Pos1', 'Pos2', 'Pos3', 'Pos4']
        colors_pos = [self.color_palette['hot'], self.color_palette['prediction'],
                     self.color_palette['success'], self.color_palette['warning']]
        
        for i in range(4):
            offset = width * i - width * 1.5
            ax2.bar(x + offset, digit_counts[i], width, label=positions[i], 
                   color=colors_pos[i], alpha=0.8)
        
        ax2.set_xlabel('Digit')
        ax2.set_ylabel('Frekuensi')
        ax2.set_title('DISTRIBUSI DIGIT PER POSISI', fontweight='bold')
        ax2.set_xticks(x)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Even/Odd ratio
        ax3 = fig.add_subplot(gs[0, 2])
        even_ratios = []
        for i in range(4):
            even_count = np.sum(digit_data_np[:, i] % 2 == 0)
            total = len(digit_data_np[:, i])
            even_ratios.append(even_count / total * 100)
        
        bars = ax3.bar(range(4), even_ratios, 
                      color=[self.color_palette['cold'] if r > 50 else self.color_palette['hot'] 
                            for r in even_ratios],
                      edgecolor='black')
        ax3.set_xlabel('Posisi')
        ax3.set_ylabel('Persentase Genap (%)')
        ax3.set_title('PERSENTASE ANGKA GENAP PER POSISI', fontweight='bold')
        ax3.set_xticks(range(4))
        ax3.set_xticklabels(['1', '2', '3', '4'])
        ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        
        # Add percentage labels
        for i, (bar, ratio) in enumerate(zip(bars, even_ratios)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{ratio:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Frequency histogram
        ax4 = fig.add_subplot(gs[1, :])
        all_counts = counts[counts > 0]
        n, bins, patches = ax4.hist(all_counts, bins=30, alpha=0.7, 
                                   color=self.color_palette['prediction'], edgecolor='black')
        
        # Color bars based on frequency
        cmap = plt.cm.viridis
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = bin_centers - min(bin_centers)
        col /= max(col)
        
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cmap(c))
        
        ax4.set_xlabel('Frekuensi Kemunculan', fontsize=11)
        ax4.set_ylabel('Jumlah Angka', fontsize=11)
        ax4.set_title('DISTRIBUSI FREKUENSI SEMUA ANGKA', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Sum of digits distribution
        ax5 = fig.add_subplot(gs[2, 0])
        digit_sums = np.sum(digit_data_np, axis=1)
        unique_sums, sum_counts = np.unique(digit_sums, return_counts=True)
        
        colors_sum = self.cmap_hot(np.linspace(0.3, 0.8, len(unique_sums)))
        ax5.bar(unique_sums, sum_counts, color=colors_sum, edgecolor='black', alpha=0.8)
        ax5.set_xlabel('Jumlah 4 Digit')
        ax5.set_ylabel('Frekuensi')
        ax5.set_title('DISTRIBUSI JUMLAH 4 DIGIT', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Unique digits per number
        ax6 = fig.add_subplot(gs[2, 1])
        unique_digits_counts = []
        for num in self.all_numbers_flat[:10000]:  # Sample for performance
            unique_digits_counts.append(len(set(str(num))))
        
        unique_counts, digit_freq = np.unique(unique_digits_counts, return_counts=True)
        colors_unique = [self.color_palette['success'] if u == 4 else 
                        self.color_palette['warning'] if u == 3 else 
                        self.color_palette['hot'] for u in unique_counts]
        
        ax6.bar(unique_counts, digit_freq, color=colors_unique, edgecolor='black', alpha=0.8)
        ax6.set_xlabel('Jumlah Digit Unik')
        ax6.set_ylabel('Frekuensi')
        ax6.set_title('DISTRIBUSI DIGIT UNIK PER ANGKA', fontweight='bold')
        ax6.set_xticks(unique_counts)
        ax6.grid(True, alpha=0.3)
        
        # Plot 7: Summary statistics
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        # Calculate statistics
        stats_text = f"""
        📊 STATISTIK DATA TOTO 4D 📊
        {'='*40}
        
        📅 Total Undian: {len(self.data):,}
        🔢 Total Angka: {len(self.all_numbers_flat):,}
        🎯 Angka Unik: {len(unique_values):,}
        
        📈 Frekuensi Rata-rata: {np.mean(counts):.2f}
        📉 Frekuensi Minimum: {np.min(counts)}
        📈 Frekuensi Maksimum: {np.max(counts)}
        
        🔥 Angka Paling Sering: {unique_values[np.argmax(counts)]} 
          ({np.max(counts)} kali)
        ❄️ Angka Paling Jarang: {unique_values[np.argmin(counts)]} 
          ({np.min(counts)} kali)
        
        ⚖️ Rata-rata Genap: {np.mean(even_ratios):.1f}%
        🎲 Rata-rata Jumlah Digit: {np.mean(digit_sums):.1f}
        """
        
        ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes,
                fontsize=11, fontfamily='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=self.color_palette['background'], 
                         alpha=0.9, edgecolor=self.color_palette['text']))
        
        # Main title
        plt.suptitle('DASHBOARD ANALISIS TOTO 4D - VISUALISASI KOMPREHENSIF', 
                    fontsize=22, fontweight='bold', color=self.color_palette['text'], y=1.02)
        
        # Footer
        plt.figtext(0.5, 0.01, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                   f'Data Points: {len(self.all_numbers_flat):,}', 
                   ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.savefig('summary_dashboard.png', dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✅ Dashboard summary disimpan sebagai 'summary_dashboard.png'")
        plt.show()
        
        return fig
    
    # ============================================
    # ANALISIS FUNCTIONS (updated with visualizations)
    # ============================================
    
    def frequency_analysis_with_predictions(self):
        """Analisis Kekerapan + 5 Predictions"""
        if len(self.all_numbers_flat) == 0:
            print("❌ Data belum diproses!")
            return [], []
        
        print("\n" + "="*60)
        print("1. ANALISIS KERAPAN + 5 PREDICTIONS")
        print("="*60)
        
        # Get frequencies
        unique_values, counts = np.unique(self.all_numbers_flat, return_counts=True)
        total_numbers = len(self.all_numbers_flat)
        
        print(f"\n📊 Statistik:")
        print(f"   • Total angka: {total_numbers:,}")
        print(f"   • Angka unik: {len(unique_values):,}")
        
        # Generate 5 predictions
        predictions = []
        
        # Get top 5 numbers
        top_indices = np.argsort(counts)[-5:][::-1]
        for pos in top_indices[:3]:
            predictions.append(unique_values[pos])
        
        # Add 2 random variations
        for _ in range(2):
            if len(unique_values) > 0:
                random_idx = np.random.randint(0, len(unique_values))
                predictions.append(unique_values[random_idx])
        
        print(f"\n🎯 5 PREDICTIONS:")
        for i, pred in enumerate(predictions[:5], 1):
            idx = np.where(unique_values == pred)[0]
            freq = counts[idx[0]] if len(idx) > 0 else 0
            color_code = f"\033[91m{pred}\033[0m" if freq > np.mean(counts) else f"\033[94m{pred}\033[0m"
            print(f"   {i}. {color_code} (muncul {freq} kali)")
        
        # Create visualization
        self.create_frequency_heatmap(20)
        
        return predictions[:5], unique_values
    
    def digit_analysis_with_predictions(self):
        """Analisis Digit + 5 Predictions"""
        if len(self.digit_data) == 0:
            print("❌ Data belum diproses!")
            return [], []
        
        print("\n" + "="*60)
        print("2. ANALISIS DIGIT + 5 PREDICTIONS")
        print("="*60)
        
        # Generate 5 predictions
        predictions = []
        
        for _ in range(5):
            suggestion = ''
            for pos in range(4):
                pos_digits = self.digit_data[:, pos]
                unique_d, digit_counts = np.unique(pos_digits, return_counts=True)
                
                # Weighted selection
                if len(unique_d) > 0:
                    probabilities = digit_counts / digit_counts.sum()
                    suggestion += str(np.random.choice(unique_d, p=probabilities))
                else:
                    suggestion += str(np.random.randint(0, 10))
            
            predictions.append(suggestion)
        
        print(f"\n🎯 5 PREDICTIONS:")
        for i, pred in enumerate(predictions[:5], 1):
            color_code = ""
            for digit in pred:
                if int(digit) % 2 == 0:
                    color_code += f"\033[94m{digit}\033[0m"  # Blue for even
                else:
                    color_code += f"\033[91m{digit}\033[0m"  # Red for odd
            print(f"   {i}. {color_code}")
        
        # Create visualization
        self.create_digit_distribution_chart()
        
        return predictions[:5], []
    
    def hot_cold_analysis_with_predictions(self, top_n=30):
        """Hot vs Cold + 5 Predictions"""
        if len(self.all_numbers_flat) == 0:
            print("❌ Data belum diproses!")
            return [], []
        
        print("\n" + "="*60)
        print("3. HOT vs COLD + 5 PREDICTIONS")
        print("="*60)
        
        unique_values, counts = np.unique(self.all_numbers_flat, return_counts=True)
        
        # Get detailed hot and cold numbers
        hot_indices = np.argsort(counts)[-top_n:][::-1]
        cold_indices = np.argsort(counts)[:top_n]
        
        hot_numbers = [(unique_values[i], counts[i]) for i in hot_indices[:10]]
        cold_numbers = [(unique_values[i], counts[i]) for i in cold_indices[:10]]
        
        print(f"\n🔥 TOP {min(10, len(hot_numbers))} HOT NUMBERS (sering keluar):")
        for i, (num, freq) in enumerate(hot_numbers[:10], 1):
            print(f"   \033[91m{i:2d}. {num}: {freq} kali\033[0m")
        
        print(f"\n❄️  TOP {min(10, len(cold_numbers))} COLD NUMBERS (jarang keluar):")
        for i, (num, freq) in enumerate(cold_numbers[:10], 1):
            print(f"   \033[94m{i:2d}. {num}: {freq} kali\033[0m")
        
        # Generate 5 predictions
        predictions = []
        
        # 2 hot numbers
        hot_indices = np.argsort(counts)[-5:][::-1]
        for pos in hot_indices[:2]:
            predictions.append(unique_values[pos])
        
        # 2 cold numbers
        if len(unique_values) > 5:
            cold_indices = np.argsort(counts)[:5]
            for pos in cold_indices[:2]:
                predictions.append(unique_values[pos])
        
        # 1 random (mix)
        if len(unique_values) > 0:
            random_idx = np.random.randint(0, len(unique_values))
            predictions.append(unique_values[random_idx])
        
        print(f"\n🎯 5 PREDICTIONS (2 Hot + 2 Cold + 1 Random):")
        for i, pred in enumerate(predictions[:5], 1):
            idx = np.where(unique_values == pred)[0]
            freq = counts[idx[0]] if len(idx) > 0 else 0
            
            # Color coding
            if i <= 2:
                color_code = f"\033[91m{pred}\033[0m"  # Red for hot
                status = "🔥 HOT"
            elif i <= 4:
                color_code = f"\033[94m{pred}\033[0m"  # Blue for cold
                status = "❄️ COLD"
            else:
                color_code = f"\033[92m{pred}\033[0m"  # Green for random
                status = "🎲 RANDOM"
            
            print(f"   {i}. {color_code} - {status} ({freq} kali)")
        
        # Return hot and cold numbers for reference
        hot_cold_info = {
            'hot': hot_numbers[:10],
            'cold': cold_numbers[:10],
            'predictions': predictions[:5]
        }
        
        # Create visualization
        self.create_hot_cold_chart(15)
        
        return predictions[:5], hot_cold_info
    
    def even_odd_analysis_with_predictions(self):
        """Genap & Ganjil + 5 Predictions"""
        if len(self.digit_data) == 0:
            print("❌ Data belum diproses!")
            return [], []
        
        print("\n" + "="*60)
        print("4. GENAP & GANJIL + 5 PREDICTIONS")
        print("="*60)
        
        predictions = []
        
        # Generate different patterns
        patterns = ['EEOO', 'EOEO', 'OOEE', 'OEOE', 'EEEE']
        
        for pattern in patterns[:5]:
            suggestion = ''
            for p in pattern:
                if p == 'E':
                    suggestion += str(np.random.choice([0, 2, 4, 6, 8]))
                else:
                    suggestion += str(np.random.choice([1, 3, 5, 7, 9]))
            predictions.append(suggestion)
        
        print(f"\n🎯 5 PREDICTIONS:")
        for i, pred in enumerate(predictions[:5], 1):
            even_count = sum(1 for d in map(int, pred) if d % 2 == 0)
            # Color code the number
            colored_pred = ""
            for digit in pred:
                if int(digit) % 2 == 0:
                    colored_pred += f"\033[94m{digit}\033[0m"  # Blue for even
                else:
                    colored_pred += f"\033[91m{digit}\033[0m"  # Red for odd
            print(f"   {i}. {colored_pred} ({even_count} genap, {4-even_count} ganjil)")
        
        # Create visualization
        self.create_even_odd_pie_chart()
        
        return predictions[:5], []
    
    # ============================================
    # PREDICTIONS POPULER ANALYSIS (updated)
    # ============================================
    
    def predictions_populer_analysis(self, all_predictions_dict):
        """Analisis Predictions Populer dari semua metode"""
        print("\n" + "="*60)
        print("ANALISIS PREDICTIONS POPULER")
        print("="*60)
        
        if not all_predictions_dict:
            print("❌ Tidak ada predictions untuk dianalisis!")
            return []
        
        # Kumpulkan semua predictions dari semua analisis
        all_preds_flat = []
        print("\n📊 Sumber Predictions:")
        
        for name, predictions in all_predictions_dict.items():
            print(f"   • {name}: {', '.join(predictions[:3])}...")
            all_preds_flat.extend(predictions)
        
        total_predictions = len(all_preds_flat)
        unique_predictions = len(set(all_preds_flat))
        
        print(f"\n📈 Statistik Predictions:")
        print(f"   • Total predictions dari semua analisis: {total_predictions}")
        print(f"   • Predictions unik: {unique_predictions}")
        print(f"   • Rata-rata duplikasi: {total_predictions/unique_predictions:.2f}x")
        
        # Hitung frekuensi kemunculan
        pred_counter = Counter(all_preds_flat)
        
        # Kategorikan predictions
        sangat_populer = []  # muncul >= 3 analisis
        populer = []        # muncul 2 analisis
        unik = []          # muncul 1 analisis
        
        for pred, count in pred_counter.items():
            if count >= 3:
                sangat_populer.append((pred, count))
            elif count == 2:
                populer.append((pred, count))
            else:
                unik.append((pred, count))
        
        # Urutkan berdasarkan frekuensi
        sangat_populer.sort(key=lambda x: x[1], reverse=True)
        populer.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🎯 PREDICTIONS SANGAT POPULER (muncul ≥3 analisis):")
        if sangat_populer:
            for pred, count in sangat_populer[:10]:  # Top 10
                print(f"   • \033[91m{pred}\033[0m: muncul di {count} analisis")
        else:
            print("   Tidak ada predictions yang muncul ≥3 kali")
        
        print(f"\n📊 PREDICTIONS POPULER (muncul 2 analisis):")
        if populer:
            for pred, count in populer[:10]:  # Top 10
                print(f"   • \033[93m{pred}\033[0m: muncul di {count} analisis")
        else:
            print("   Tidak ada predictions yang muncul 2 kali")
        
        # Rekomendasi akhir
        print(f"\n💡 REKOMENDASI AKHIR:")
        
        rekomendasi_akhir = []
        
        # Prioritaskan predictions sangat populer
        if sangat_populer:
            print(f"   ⭐ REKOMENDASI TINGGI (prioritas utama):")
            for pred, count in sangat_populer[:5]:
                rekomendasi_akhir.append(pred)
                print(f"      • \033[91m{pred}\033[0m - {count} analisis")
        
        # Tambahkan predictions populer jika kurang dari 5
        if len(rekomendasi_akhir) < 5 and populer:
            print(f"\n   ⭐ REKOMENDASI SEDANG (tambahan):")
            for pred, count in populer[:5-len(rekomendasi_akhir)]:
                if pred not in rekomendasi_akhir:
                    rekomendasi_akhir.append(pred)
                    print(f"      • \033[93m{pred}\033[0m - {count} analisis")
        
        # Tambahkan random dari unik jika masih kurang
        if len(rekomendasi_akhir) < 5 and unik:
            print(f"\n   ⭐ REKOMENDASI DASAR (pengisian):")
            import random
            selected = random.sample(unik, min(5-len(rekomendasi_akhir), len(unik)))
            for pred, count in selected:
                rekomendasi_akhir.append(pred)
                print(f"      • \033[92m{pred}\033[0m - {count} analisis")
        
        # Tampilkan rekomendasi akhir
        print(f"\n🎯 5 REKOMENDASI AKHIR (berdasarkan konsensus):")
        for i, pred in enumerate(rekomendasi_akhir[:5], 1):
            count = pred_counter.get(pred, 0)
            confidence = "Tinggi" if count >= 3 else "Sedang" if count == 2 else "Dasar"
            
            # Color based on confidence
            if confidence == "Tinggi":
                color = "\033[91m"  # Red
            elif confidence == "Sedang":
                color = "\033[93m"  # Yellow
            else:
                color = "\033[92m"  # Green
            
            print(f"   {i}. {color}{pred}\033[0m (confidence: {confidence}, muncul: {count} analisis)")
        
        # Create visualization
        self.create_predictions_visualization(all_predictions_dict)
        
        return rekomendasi_akhir[:5]
    
    # ============================================
    # MAIN ANALYSIS FUNCTION (updated)
    # ============================================
    
    def run_all_analyses_with_predictions(self, export_mode=False):
        """Jalankan semua analisis"""
        if export_mode:
            # In export mode, just run analyses without interaction
            all_predictions = {}
            
            analyses = [
                ("1. Analisis Kekerapan", self.frequency_analysis_with_predictions),
                ("2. Analisis Digit", self.digit_analysis_with_predictions),
                ("3. Hot vs Cold Number", lambda: self.hot_cold_analysis_with_predictions(30)),
                ("4. Analisis Genap & Ganjil", self.even_odd_analysis_with_predictions),
                ("5. Analisis Jumlah Digit", self.digit_analysis_with_predictions),
                ("6. Analisis Ulangan Digit", self.digit_analysis_with_predictions),
                ("7. Analisis Corak", self.digit_analysis_with_predictions),
                ("8. Analisis Posisi Hadiah", self.digit_analysis_with_predictions),
                ("9. Analisis Sliding Window", lambda: self.sliding_window_analysis_with_predictions(20)),
                ("10. Statistik Komprehensif", self.digit_analysis_with_predictions),
                ("11. Analisis Nombor Paling Jarang Keluar", self.digit_analysis_with_predictions)
            ]
            
            for name, analysis_func in analyses:
                try:
                    print(f"\n{name}")
                    print("-" * 60)
                    predictions, _ = analysis_func()
                    all_predictions[name] = predictions
                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue
            
            # Analisis Predictions Populer
            print("\n" + "="*60)
            print("12. ANALISIS PREDICTIONS POPULER")
            print("="*60)
            
            rekomendasi_akhir = self.predictions_populer_analysis(all_predictions)
            
            # Create summary dashboard
            print("\n" + "="*60)
            print("📊 MEMBUAT DASHBOARD SUMMARY")
            print("="*60)
            self.create_summary_dashboard()
            
            return all_predictions
        
        else:
            # Interactive mode
            print("\n" + "="*100)
            print("🚀 SEMUA ANALISIS DENGAN PREDICTIONS")
            print("="*100)
            
            all_predictions = {}
            
            analyses = [
                ("1. Analisis Kekerapan", self.frequency_analysis_with_predictions),
                ("2. Analisis Digit", self.digit_analysis_with_predictions),
                ("3. Hot vs Cold Number", lambda: self.hot_cold_analysis_with_predictions(30)),
                ("4. Analisis Genap & Ganjil", self.even_odd_analysis_with_predictions),
                ("5. Analisis Jumlah Digit", self.digit_analysis_with_predictions),
                ("6. Analisis Ulangan Digit", self.digit_analysis_with_predictions),
                ("7. Analisis Corak", self.digit_analysis_with_predictions),
                ("8. Analisis Posisi Hadiah", self.digit_analysis_with_predictions),
                ("9. Analisis Sliding Window", lambda: self.sliding_window_analysis_with_predictions(20)),
                ("10. Statistik Komprehensif", self.digit_analysis_with_predictions),
                ("11. Analisis Nombor Paling Jarang Keluar", self.even_odd_analysis_with_predictions)
            ]
            

            for name, analysis_func in analyses:
                print(f"\n▶️  {name}")
                print("-" * 100)
                try:
                    predictions, _ = analysis_func()
                    all_predictions[name] = predictions
                    
                    print(f"\n   📋 Predictions: {', '.join(predictions)}")
                    
                    # Ask if user wants to see visualization
                    if not export_mode:
                        viz_choice = input("\n   👁️  Tampilkan visualisasi? (y/n): ").strip().lower()
                        if viz_choice == 'y':
                            # Create specific visualization based on analysis
                            if name.startswith("1."):
                                self.create_frequency_heatmap(20)
                            elif name.startswith("2."):
                                self.create_digit_distribution_chart()
                            elif name.startswith("3."):
                                self.create_hot_cold_chart(15)
                            elif name.startswith("4."):
                                self.create_even_odd_pie_chart()
                    
                    input("\n   ⏸️  Tekan Enter untuk lanjut...")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue
            
            # Analisis Predictions Populer
            print("\n" + "="*100)
            print("📊 ANALISIS PREDICTIONS POPULER")
            print("="*100)
            
            rekomendasi_akhir = self.predictions_populer_analysis(all_predictions)
            
            # Create summary dashboard
            print("\n" + "="*100)
            print("📊 MEMBUAT DASHBOARD SUMMARY")
            print("="*100)
            self.create_summary_dashboard()
            
            return all_predictions, rekomendasi_akhir


def main():
    print("\n" + "="*100)
    print("🎯 TOTO 4D - ALL ANALYSES WITH PREDICTIONS & VISUALIZATIONS")
    print("="*100)
    
    analyzer = TOTO4DAnalyzer()
    
    while True:
        print("\n" + "="*100)
        print("🏠 MENU UTAMA DENGAN VISUALISASI")
        print("="*100)
        
        print("\n📊 ANALISIS DASAR:")
        print("  1. 📂 Muat data")
        print("  2. 🚀 Jalankan SEMUA analisis + predictions")
        print("  3. 1️. Analisis Kekerapan + 5 predictions")
        print("  4. 2️. Analisis Digit + 5 predictions")
        print("  5. 3️. Hot vs Cold + 5 predictions")
        print("  6. 4️. Genap & Ganjil + 5 predictions")
        print("  7. 5️. Jumlah Digit + 5 predictions")
        print("  8. 6️. Ulangan Digit + 5 predictions")
        
        print("\n📈 VISUALISASI LANJUTAN:")
        print("  9. 📊 Tampilkan semua visualisasi")
        print("  10. 🔥 Heatmap frekuensi angka")
        print("  11. 🎯 Chart distribusi digit")
        print("  12. ❄️ Hot vs Cold visualization")
        print("  13. ⚖️ Even/Odd pie charts")
        print("  14. 📋 Dashboard summary lengkap")
        
        print("\n🎯 FITUR LAIN:")
        print("  15. 📊 Analisis Predictions Populer")
        print("  16. 💾 Ekspor laporan lengkap")
        print("  17. ❌ Keluar")
        
        choice = input("\nPilihan (1-17): ").strip()
        
        if choice == '1':
            file_path = input("Path file data: ").strip()
            if file_path and os.path.exists(file_path):
                success = analyzer.load_data_large(file_path)
                if success:
                    print(f"✅ Data berhasil dimuat: {len(analyzer.data):,} baris")
                else:
                    print("❌ Gagal memuat data")
            else:
                print("❌ File tidak ditemukan")
        
        elif choice == '2':
            if analyzer.data is not None:
                analyzer.run_all_analyses_with_predictions(export_mode=False)
            else:
                print("❌ Muat data dulu!")
        
        elif choice in ['3', '4', '5', '6', '7', '8']:
            if analyzer.data is not None:
                analysis_map = {
                    '3': ("Analisis Kekerapan", analyzer.frequency_analysis_with_predictions),
                    '4': ("Analisis Digit", analyzer.digit_analysis_with_predictions),
                    '5': ("Hot vs Cold", lambda: analyzer.hot_cold_analysis_with_predictions(30)),
                    '6': ("Genap & Ganjil", analyzer.even_odd_analysis_with_predictions),
                    '7': ("Jumlah Digit", analyzer.digit_analysis_with_predictions),
                    '8': ("Ulangan Digit", analyzer.digit_analysis_with_predictions)
                }
                
                if choice in analysis_map:
                    name, func = analysis_map[choice]
                    print(f"\n▶️  {name}")
                    print("="*60)
                    func()
            else:
                print("❌ Muat data dulu!")
        
        elif choice == '9':
            if analyzer.data is not None:
                print("\n" + "="*100)
                print("📊 MENAMPILKAN SEMUA VISUALISASI")
                print("="*100)
                
                print("\n1. Heatmap frekuensi angka...")
                analyzer.create_frequency_heatmap(20)
                
                print("\n2. Chart distribusi digit...")
                analyzer.create_digit_distribution_chart()
                
                print("\n3. Hot vs Cold visualization...")
                analyzer.create_hot_cold_chart(15)
                
                print("\n4. Even/Odd pie charts...")
                analyzer.create_even_odd_pie_chart()
                
                print("\n5. Dashboard summary...")
                analyzer.create_summary_dashboard()
                
                print("✅ Semua visualisasi telah dibuat dan disimpan!")
            else:
                print("❌ Muat data dulu!")
        
        elif choice == '10':
            if analyzer.data is not None:
                analyzer.create_frequency_heatmap(20)
            else:
                print("❌ Muat data dulu!")
        
        elif choice == '11':
            if analyzer.data is not None:
                analyzer.create_digit_distribution_chart()
            else:
                print("❌ Muat data dulu!")
        
        elif choice == '12':
            if analyzer.data is not None:
                analyzer.create_hot_cold_chart(15)
            else:
                print("❌ Muat data dulu!")
        
        elif choice == '13':
            if analyzer.data is not None:
                analyzer.create_even_odd_pie_chart()
            else:
                print("❌ Muat data dulu!")
        
        elif choice == '14':
            if analyzer.data is not None:
                analyzer.create_summary_dashboard()
            else:
                print("❌ Muat data dulu!")
        
        elif choice == '15':
            if analyzer.data is not None:
                print("\n" + "="*100)
                print("📊 ANALISIS PREDICTIONS POPULER")
                print("="*100)
                
                # Jalankan semua analisis terlebih dahulu
                print("\n⏳ Menjalankan semua analisis untuk mengumpulkan predictions...")
                all_predictions, _ = analyzer.run_all_analyses_with_predictions(export_mode=True)
                
                # Tampilkan analisis predictions populer
                input("\n⏸️  Tekan Enter untuk melihat Analisis Predictions Populer...")
                analyzer.predictions_populer_analysis(all_predictions)
            else:
                print("❌ Muat data dulu!")
        
        elif choice == '16':
            if analyzer.data is not None:
                filename = input("Nama file output (default: predictions_report.txt): ").strip()
                if not filename:
                    filename = "predictions_report.txt"
                
                print(f"⏳ Membuat laporan '{filename}'...")
                
                # Save original stdout
                original_stdout = sys.stdout
                
                try:
                    # Create output file
                    with open(filename, 'w', encoding='utf-8') as f:
                        # Redirect stdout to file
                        sys.stdout = f
                        
                        print("="*100)
                        print("LAPORAN PREDICTIONS TOTO 4D DENGAN VISUALISASI")
                        print("="*100)
                        print(f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"Data: {len(analyzer.data):,} undian")
                        print(f"Total angka: {len(analyzer.all_numbers_flat):,}")
                        print("="*100)
                        
                        # Run analyses in export mode
                        analyzer.run_all_analyses_with_predictions(export_mode=True)
                    
                    # Restore stdout
                    sys.stdout = original_stdout
                    
                    print(f"✅ Laporan berhasil diekspor ke: {filename}")
                    print(f"📄 Ukuran file: {os.path.getsize(filename):,} bytes")
                    print(f"📊 Visualisasi disimpan sebagai file PNG dan HTML")
                    
                except Exception as e:
                    # Make sure stdout is restored
                    sys.stdout = original_stdout
                    print(f"❌ Error mengekspor laporan: {e}")
                    
            else:
                print("❌ Silakan muat data terlebih dahulu!")
        
        elif choice == '17':
            print("\n" + "="*100)
            print("👋 TERIMA KASIH! Semoga beruntung! 🍀")
            print("="*100)
            break
        
        else:
            print("❌ Pilihan tidak valid!")
        
        if choice != '17':
            input("\n⏸️  Tekan Enter untuk kembali...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Program dihentikan")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
