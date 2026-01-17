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
warnings.filterwarnings('ignore')

class TOTO4DAnalyzer:
    def __init__(self, data_file=None, chunk_size=10000):
        """Initialize the TOTO 4D Analyzer"""
        self.data = None
        self.all_numbers_flat = []
        self.digit_data = []
        self.recent_data = None
        self.chunk_size = chunk_size
        
        if data_file:
            self.load_data_large(data_file)
    
    def load_data_large(self, file_path):
        """Load historical data"""
        try:
            if not os.path.exists(file_path):
                print(f"❌ File '{file_path}' Not Found!")
                return False
            
            print(f"📂 Load Data From : {file_path}")
            
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
            
            print(f"✅ Data Loaded Successfull : {len(self.data):,} Record")
            
            # Preprocessing
            success = self.preprocess_data_large()
            return success
            
        except Exception as e:
            print(f"❌ Error Loaded Data: {e}")
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
                
                print(f"📊 Total Voting Results : {len(self.data):,}")
                
                self.recent_data = self.data.tail(1000).copy() if len(self.data) >= 1000 else self.data.copy()
            else:
                print("❌ Columns 'Draw_Date' Not Found!")
                return False
            
            # Process numbers
            number_columns = [col for col in self.data.columns if col != 'Draw_Date']
            
            print(f"🔢 Processing {len(number_columns)} Columns...")
            
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
                    print(f"   Progress: {(end_idx/len(self.data)*60):.1f}%")
            
            print(f"✅ Total Number Processed : {len(self.all_numbers_flat):,}")
            
            # Convert to numpy for efficiency
            self.all_numbers_flat = np.array(self.all_numbers_flat, dtype='U4')
            self.digit_data = np.array(self.digit_data, dtype=np.uint8)
            
            return True
            
        except Exception as e:
            print(f"❌ Error Preprocessing Data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ============================================
    # analysis FUNCTIONS (simplified for export)
    # ============================================
    
    def frequency_analysis_with_predictions(self):
        """analysis Kekerapan + 5 Predictions"""
        if len(self.all_numbers_flat) == 0:
            print("❌ The Data Has Not Been Processed!")
            return [], []
        
        print("\n" + "="*60)
        print("1. ANALYSES FREQUENCY + 5 PREDICTIONS")
        print("="*60)
        
        # Get frequencies
        unique_values, counts = np.unique(self.all_numbers_flat, return_counts=True)
        total_numbers = len(self.all_numbers_flat)
        
        print(f"\n📊 Statistics :")
        print(f"   • Total Numbers: {total_numbers:,}")
        print(f"   • Unique Numbers : {len(unique_values):,}")
        
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
            print(f"   {i}. {pred} (appear {freq} times)")
        
        return predictions[:5], unique_values
    
    def digit_analysis_with_predictions(self):
        """analysis Digit + 5 Predictions"""
        if len(self.digit_data) == 0:
            print("❌ Data Not Yet Processed!")
            return [], []
        
        print("\n" + "="*60)
        print("2. ANALYSES DIGIT + 5 PREDICTIONS")
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
            print(f"   {i}. {pred}")
        
        return predictions[:5], []
    
    def hot_cold_analysis_with_predictions(self, top_n=30):
        """Hot vs Cold + 5 Predictions"""
        if len(self.all_numbers_flat) == 0:
            print("❌ Data Not Yet Processed!")
            return [], []  # Ubah dari 3 menjadi 2 return values
        
        print("\n" + "="*60)
        print("3. ANALYSES HOT vs COLD NUMBER + 5 PREDICTIONS")
        print("="*60)
        
        unique_values, counts = np.unique(self.all_numbers_flat, return_counts=True)
        
        # Get detailed hot and cold numbers
        hot_indices = np.argsort(counts)[-top_n:][::-1]
        cold_indices = np.argsort(counts)[:top_n]
        
        hot_numbers = [(unique_values[i], counts[i]) for i in hot_indices[:10]]
        cold_numbers = [(unique_values[i], counts[i]) for i in cold_indices[:10]]
        
        print(f"\n🔥 TOP {min(10, len(hot_numbers))} HOT NUMBERS (go out often):")
        for i, (num, freq) in enumerate(hot_numbers[:10], 1):
            print(f"   {i:2d}. {num}: {freq} times")
        
        print(f"\n❄️  TOP {min(10, len(cold_numbers))} COLD NUMBERS (rarely go out):")
        for i, (num, freq) in enumerate(cold_numbers[:10], 1):
            print(f"   {i:2d}. {num}: {freq} times")
        
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
            
            # Determine status based on position in predictions
            if i <= 2:
                status = "🔥 HOT"
            elif i <= 4:
                status = "❄️ COLD"
            else:
                status = "🎲 RANDOM"
            
            print(f"   {i}. {pred} - {status} ({freq} times)")
        
        # Return hot and cold numbers for reference
        hot_cold_info = {
            'hot': hot_numbers[:10],
            'cold': cold_numbers[:10],
            'predictions': predictions[:5]
        }
        
        return predictions[:5], hot_cold_info  # Kembalikan 2 values saja
    
    def even_odd_analysis_with_predictions(self):
        """Genap & Ganjil + 5 Predictions"""
        if len(self.digit_data) == 0:
            print("❌ Data Not Yet Processed!")
            return [], []
        
        print("\n" + "="*60)
        print("4. ANALYSIS OF EVEN AND ODD NUMBERS + 5 PREDICTIONS")
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
            print(f"   {i}. {pred} ({even_count} even, {4-even_count} odd)")
        
        return predictions[:5], []
    
    def digit_sum_analysis_with_predictions(self):
        """Jumlah Digit + 5 Predictions"""
        if len(self.digit_data) == 0:
            print("❌ Data Not Yet Processed!")
            return [], []
        
        print("\n" + "="*60)
        print("5. ANALYSES OF THE SUM OF DIGITST + 5 PREDICTIONS")
        print("="*60)
        
        # Calculate common sums
        digit_sums = np.sum(self.digit_data, axis=1)
        unique_sums, sum_counts = np.unique(digit_sums, return_counts=True)
        
        predictions = []
        
        # Generate numbers with common sums
        common_sum_indices = np.argsort(sum_counts)[-3:][::-1]
        for pos in common_sum_indices:
            target_sum = unique_sums[pos]
            
            # Generate number with this sum
            for attempt in range(50):
                digits = [np.random.randint(0, 10) for _ in range(4)]
                if sum(digits) == target_sum:
                    suggestion = ''.join(str(d) for d in digits)
                    if suggestion not in predictions:
                        predictions.append(suggestion)
                        break
        
        # Fill remaining
        while len(predictions) < 5:
            digits = [np.random.randint(0, 10) for _ in range(4)]
            suggestion = ''.join(str(d) for d in digits)
            if suggestion not in predictions:
                predictions.append(suggestion)
        
        print(f"\n🎯 5 PREDICTIONS:")
        for i, pred in enumerate(predictions[:5], 1):
            digit_sum = sum(int(d) for d in pred)
            print(f"   {i}. {pred} (total: {digit_sum})")
        
        return predictions[:5], []
    
    def digit_repetition_analysis_with_predictions(self):
        """Ulangan Digit + 5 Predictions"""
        if len(self.digit_data) == 0:
            print("❌ Data Not Yet Processed!")
            return [], []
        
        print("\n" + "="*60)
        print("6. ANALYSIS DIGIT REPETITION + 5 PREDICTIONS")
        print("="*60)
        
        predictions = []
        
        # Different repetition patterns
        # 1. All different
        digits = []
        while len(digits) < 4:
            digit = np.random.randint(0, 10)
            if digit not in digits:
                digits.append(digit)
        predictions.append(''.join(str(d) for d in digits))
        
        # 2. 2 same, 2 different
        same_digit = np.random.randint(0, 10)
        other_digits = []
        while len(other_digits) < 2:
            digit = np.random.randint(0, 10)
            if digit != same_digit and digit not in other_digits:
                other_digits.append(digit)
        predictions.append(f"{same_digit}{same_digit}{other_digits[0]}{other_digits[1]}")
        
        # 3. 2 pairs
        digit1 = np.random.randint(0, 10)
        digit2 = np.random.randint(0, 10)
        while digit2 == digit1:
            digit2 = np.random.randint(0, 10)
        predictions.append(f"{digit1}{digit1}{digit2}{digit2}")
        
        # 4. 3 same
        main_digit = np.random.randint(0, 10)
        different_digit = np.random.randint(0, 10)
        while different_digit == main_digit:
            different_digit = np.random.randint(0, 10)
        predictions.append(f"{main_digit}{main_digit}{main_digit}{different_digit}")
        
        # 5. Random
        predictions.append(f"{np.random.randint(0, 10000):04d}")
        
        print(f"\n🎯 5 PREDICTIONS:")
        for i, pred in enumerate(predictions[:5], 1):
            unique_digits = len(set(pred))
            print(f"   {i}. {pred} ({unique_digits} digit unique)")
        
        return predictions[:5], {}
    
    def pattern_analysis_with_predictions(self):
        """Corak + 5 Predictions"""
        if len(self.digit_data) == 0:
            print("❌ Data Not Yet Processed!")
            return [], []
        
        print("\n" + "="*60)
        print("7. ANALYSIS PATTERN + 5 PREDICTIONS")
        print("="*60)
        
        predictions = []
        
        # 1. Sequential
        start = np.random.randint(0, 7)
        predictions.append(''.join(str((start + i) % 10) for i in range(4)))
        
        # 2. Mirror
        first = np.random.randint(0, 10)
        second = np.random.randint(0, 10)
        predictions.append(f"{first}{second}{second}{first}")
        
        # 3. Palindrome
        a = np.random.randint(0, 10)
        b = np.random.randint(0, 10)
        predictions.append(f"{a}{b}{b}{a}")
        
        # 4. Alternating
        pattern = np.random.choice(['EOEO', 'OEOE'])
        suggestion = ''
        for p in pattern:
            if p == 'E':
                suggestion += str(np.random.choice([0, 2, 4, 6, 8]))
            else:
                suggestion += str(np.random.choice([1, 3, 5, 7, 9]))
        predictions.append(suggestion)
        
        # 5. Arithmetic
        start = np.random.randint(0, 7)
        diff = np.random.randint(1, 4)
        predictions.append(''.join(str((start + i * diff) % 10) for i in range(4)))
        
        print(f"\n🎯 5 PREDICTIONS:")
        pattern_names = {
            0: "Sequentially",
            1: "Mirror",
            2: "Palindrome,",
            3: "Interspersed",
            4: "Arithmetic"
        }
        for i, pred in enumerate(predictions[:5], 1):
            print(f"   {i}. {pred} ({pattern_names.get(i-1, 'General')})")
        
        return predictions[:5], {}
    
    def prize_position_analysis_with_predictions(self):
        """Posisi Hadiah + 5 Predictions"""
        if self.data is None:
            print("❌ Data Not Yet Processed!")
            return [], []
        
        print("\n" + "="*60)
        print("8. ANALYSIS OF PRIZE POSITION + 5 PREDICTIONS")
        print("="*60)
        
        number_columns = [col for col in self.data.columns if col != 'Draw_Date']
        
        predictions = []
        
        # Generate based on column patterns
        for _ in range(5):
            suggestion = ''
            for col_idx in range(4):
                if col_idx < len(number_columns):
                    col = number_columns[col_idx]
                    # Get sample from this column
                    sample = self.data[col].dropna().sample(min(100, len(self.data)))
                    if len(sample) > 0:
                        num_str = str(sample.iloc[0]).strip().zfill(4)[:4]
                        if num_str.isdigit() and len(num_str) == 4:
                            suggestion += num_str[col_idx % 4]
                        else:
                            suggestion += str(np.random.randint(0, 10))
                    else:
                        suggestion += str(np.random.randint(0, 10))
                else:
                    suggestion += str(np.random.randint(0, 10))
            
            predictions.append(suggestion)
        
        print(f"\n🎯 5 PREDICTIONS:")
        for i, pred in enumerate(predictions[:5], 1):
            print(f"   {i}. {pred}")
        
        return predictions[:5], []
    
    def sliding_window_analysis_with_predictions(self, window_size=20):
        """Sliding Window + 5 Predictions"""
        if self.data is None or len(self.data) < window_size:
            print(f"❌ Not Enough Data")
            return [], []
        
        print("\n" + "="*60)
        print(f"9. ANALYSES SLIDING WINDOW ({window_size}) + 5 PREDICTIONS")
        print("="*60)
        
        number_columns = [col for col in self.data.columns if col != 'Draw_Date']
        
        # Get recent data
        recent_data = self.data.tail(window_size)
        recent_numbers = []
        
        for _, row in recent_data.iterrows():
            for col in number_columns:
                num_str = str(row[col]).strip().zfill(4)[:4]
                if num_str.isdigit():
                    recent_numbers.append(num_str)
        
        predictions = []
        
        if recent_numbers:
            unique_recent, recent_counts = np.unique(recent_numbers, return_counts=True)
            
            # Add trending numbers
            if len(unique_recent) > 0:
                top_indices = np.argsort(recent_counts)[-3:][::-1]
                for pos in top_indices:
                    if len(predictions) < 3:
                        predictions.append(unique_recent[pos])
        
        # Fill remaining
        while len(predictions) < 5:
            predictions.append(f"{np.random.randint(0, 10000):04d}")
        
        print(f"\n🎯 5 PREDICTIONS:")
        for i, pred in enumerate(predictions[:5], 1):
            print(f"   {i}. {pred}")
        
        return predictions[:5], []
    
    def statistics_analysis_with_predictions(self):
        """Statistik + 5 Predictions"""
        if len(self.all_numbers_flat) == 0:
            print("❌ Data Not Yet Processed!")
            return [], {}
        
        print("\n" + "="*60)
        print("10. ANALYSIS COMPREHENSIVE STATISTICAL + 5 PREDICTIONS")
        print("="*60)
        
        total_draws = len(self.data)
        total_numbers = len(self.all_numbers_flat)
        
        print(f"\n📊 Basic Statistics:")
        print(f"   • Vote: {total_draws:,}")
        print(f"   • Numbers: {total_numbers:,}")
        
        predictions = []
        
        # Generate based on statistics
        for _ in range(5):
            suggestion = ''
            for pos in range(4):
                pos_digits = self.digit_data[:, pos]
                unique_d, digit_counts = np.unique(pos_digits, return_counts=True)
                
                if len(unique_d) > 0:
                    # Weight towards common digits
                    probabilities = digit_counts / digit_counts.sum()
                    suggestion += str(np.random.choice(unique_d, p=probabilities))
                else:
                    suggestion += str(np.random.randint(0, 10))
            
            predictions.append(suggestion)
        
        print(f"\n🎯 5 PREDICTIONS:")
        for i, pred in enumerate(predictions[:5], 1):
            digit_sum = sum(int(d) for d in pred)
            even_count = sum(1 for d in map(int, pred) if d % 2 == 0)
            print(f"   {i}. {pred} (total: {digit_sum}, even: {even_count})")
        
        return predictions[:5], {}
    
    def new_numbers_analysis_with_predictions(self):
        """Nombor Paling Jarang Keluar (Cold Numbers) + 5 Predictions"""
        if len(self.all_numbers_flat) == 0:
            print("❌ Data Not Yet Processed!")
            return [], []
        
        print("\n" + "="*60)
        print("11. ANALYSIS OF THE RAREST NUMBERS + 5 PREDICTIONS")
        print("="*60)
        
        # Hitung frekuensi semua angka
        unique_values, counts = np.unique(self.all_numbers_flat, return_counts=True)
        
        if len(unique_values) == 0:
            print("❌ No Frequency Data Available!")
            return [], []
        
        # Ambil 5 angka dengan frekuensi terendah
        cold_indices = np.argsort(counts)[:5]  # Ambil 5 terbawah
        
        predictions = []
        cold_numbers_info = []
        
        for idx in cold_indices:
            number = unique_values[idx]
            freq = counts[idx]
            predictions.append(number)
            cold_numbers_info.append((number, freq))
        
        print(f"\n🎯 5 RAREEST NUMBER OUT:")
        for i, (number, freq) in enumerate(cold_numbers_info, 1):
            # Tampilkan juga persentase kemunculan
            percentage = (freq / len(self.all_numbers_flat)) * 100 if len(self.all_numbers_flat) > 0 else 0
            
            # Tentukan status cold level
            if freq == 1:
                status = "❄️❄️❄️ (ICE)"
            elif freq == 2:
                status = "❄️❄️ (COLD)"
            else:
                status = "❄️ (CHILLY)"
            
            print(f"   {i}. {number} - {freq} times ({percentage:.4f}%) {status}")
        
        # Tampilkan statistik tambahan
        print(f"\n📊 Statistics Cold Numbers:")
        print(f"   • Total Unique Number: {len(unique_values):,}")
        print(f"   • Average Frequency: {np.mean(counts):.2f} times")
        print(f"   • Lowest Frequency: {np.min(counts)} times")
        print(f"   • Highest Frequency: {np.max(counts)} times")
        
        # Berikan saran berdasarkan cold numbers
        print(f"\n💡 Strategy Advice:")
        print(f"   • Number {cold_numbers_info[0][0]} is likely the rarest out")
        print(f"   • Consider for a combination with a hot number (frequently out)")
        print(f"   • Use in specific patterns (even/odd, number of digits)")
        
        return predictions[:5], cold_numbers_info
    
    def predictions_populer_analysis(self, all_predictions_dict):
        """analysis Predictions Populer dari semua metode"""
        #print("\n" + "="*60)
        #print("analysis PREDICTIONS POPULER")
        #print("="*60)
        
        if not all_predictions_dict:
            print("❌ There are No Predictions to Analyze!")
            return []
        
        # Kumpulkan semua predictions dari semua analysis
        all_preds_flat = []
        print("\n📊 Source Predictions:")
        
        for name, predictions in all_predictions_dict.items():
            print(f"   • {name}: {', '.join(predictions[:3])}...")
            all_preds_flat.extend(predictions)
        
        total_predictions = len(all_preds_flat)
        unique_predictions = len(set(all_preds_flat))
        
        print(f"\n📈 Statistics Predictions:")
        print(f"   • Total predictions from all Analysis: {total_predictions}")
        print(f"   • Unique Predictions: {unique_predictions}")
        print(f"   • Average duplication: {total_predictions/unique_predictions:.2f}x")
        
        # Hitung frekuensi kemunculan
        pred_counter = Counter(all_preds_flat)
        
        # Kategorikan predictions
        sangat_populer = []  # muncul >= 3 analysis
        populer = []        # muncul 2 analysis
        unique = []          # muncul 1 analysis
        
        for pred, count in pred_counter.items():
            if count >= 3:
                sangat_populer.append((pred, count))
            elif count == 2:
                populer.append((pred, count))
            else:
                unique.append((pred, count))
        
        # Urutkan berdasarkan frekuensi
        sangat_populer.sort(key=lambda x: x[1], reverse=True)
        populer.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🎯 VERY POPULAR PREDICTIONS (appear ≥3 analyses):")
        if sangat_populer:
            for pred, count in sangat_populer[:10]:  # Top 10
                print(f"   • {pred}: appears in {count} analysis")
        else:
            print("   There are no predictions that appear ≥3 times")
        
        print(f"\n📊 POPULAR PREDICTIONS (appear 2 analyses):")
        if populer:
            for pred, count in populer[:10]:  # Top 10
                print(f"   • {pred}: appear in {count} analysis")
        else:
            print("   There are no predictions that appear 2 times")
        
        # Rekomendasi akhir
        print(f"\n💡 FINAL RECOMMENDATION:")
        
        rekomendasi_akhir = []
        
        # Prioritaskan predictions sangat populer
        if sangat_populer:
            print(f"   ⭐ HIGH RECOMMENDATION (Top Priority):")
            for pred, count in sangat_populer[:5]:
                rekomendasi_akhir.append(pred)
                print(f"      • {pred} - {count} analysis")
        
        # Tambahkan predictions populer jika kurang dari 5
        if len(rekomendasi_akhir) < 5 and populer:
            print(f"\n   ⭐ MEDIUM RECOMMENDATION (Extra):")
            for pred, count in populer[:5-len(rekomendasi_akhir)]:
                if pred not in rekomendasi_akhir:
                    rekomendasi_akhir.append(pred)
                    print(f"      • {pred} - {count} analysis")
        
        # Tambahkan random dari unique jika masih kurang
        if len(rekomendasi_akhir) < 5 and unique:
            print(f"\n   ⭐ BASIC RECOMMENDATION (Filling):")
            import random
            selected = random.sample(unique, min(5-len(rekomendasi_akhir), len(unique)))
            for pred, count in selected:
                rekomendasi_akhir.append(pred)
                print(f"      • {pred} - {count} analysis")
        
        # Tampilkan rekomendasi akhir
        print(f"\n🎯 5 FINAL RECOMMENDATION (Based on Consensus):")
        for i, pred in enumerate(rekomendasi_akhir[:5], 1):
            count = pred_counter.get(pred, 0)
            confidence = "High" if count >= 3 else "medium" if count == 2 else "Base"
            print(f"   {i}. {pred} (confidence: {confidence}, appear: {count} analysis)")
        
        return rekomendasi_akhir[:5]
    
    def run_all_analyses_with_predictions(self, export_mode=False):
        """Jalankan semua analysis"""
        if export_mode:
            # In export mode, just run analyses without interaction
            all_predictions = {}
            
            analyses = [
                ("1. Analysis Frequency", self.frequency_analysis_with_predictions),
                ("2. Analysis Digit", self.digit_analysis_with_predictions),
                ("3. Analysis Hot vs Cold Number", lambda: self.hot_cold_analysis_with_predictions(30)),
                ("4. Analysis of Even & Odd Numbers", self.even_odd_analysis_with_predictions),
                ("5. Analysis of The Sum of Digitst", self.digit_sum_analysis_with_predictions),
                ("6. Analysis Digit Repetition", self.digit_repetition_analysis_with_predictions),
                ("7. Analysis Pattern", self.pattern_analysis_with_predictions),
                ("8. Analysis of Prize Position", self.prize_position_analysis_with_predictions),
                ("9. Analysis Sliding Window", lambda: self.sliding_window_analysis_with_predictions(20)),
                ("10. Analysis Comprehensive Statistical", self.statistics_analysis_with_predictions),
                ("11. Analysis of the Rarest Numbers", self.new_numbers_analysis_with_predictions)
            ]
            
            for name, analysis_func in analyses:
                try:
                    #print(f"\n{name}")
                    #print("-" * 60)
                    predictions, _ = analysis_func()
                    all_predictions[name] = predictions
                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue
            
            # analysis Predictions Populer
            print("\n" + "="*60)
            print("12. ⭐ FINAL ANALYSIS PREDICTIONS POPULER ⭐")
            print("        ⭐ BASED ON EXISTING ANALYSIS ⭐")
            print("="*60)
            
            rekomendasi_akhir = self.predictions_populer_analysis(all_predictions)
            
            # Ringkasan akhir
            print("\n" + "="*60)
            print("FINAL SUMMARY OF PREDICTIONS")
            print("="*60)
            
            all_preds_flat = []
            for predictions in all_predictions.values():
                all_preds_flat.extend(predictions)
            
            if all_preds_flat:
                pred_counter = Counter(all_preds_flat)
                print("\n📊 Overall Statistics of Predictions:")
                print(f"   • Total predictions from all Analysis: {len(all_preds_flat)}")
                print(f"   • Unique Predictions: {len(pred_counter)}")
                
                print("\n🏆 TOP 10 PREDICTIONS BASED ON FREQUENCY:")
                for pred, count in pred_counter.most_common(10):
                    confidence = "🔥" * min(count, 3)
                    print(f"   • {pred}: {count} analysis {confidence}")
            print("\n" + "="*60)
            print("🎯 GOOD LUCK HE-HE-HE-HE ⭐")
            print("="*60)
            # Tampilkan rekomendasi akhir
            #if rekomendasi_akhir:
            #    print(f"\n🎯 5 FINAL RECOMMENDATION FOR YOU:")
            #    for i, pred in enumerate(rekomendasi_akhir[:5], 1):
            #        print(f"   {i}. {pred}")
            
            return all_predictions
        
        else:
            # Interactive mode
            print("\n" + "="*60)
            print("🚀 ALL ANALYSES WITH PREDICTIONS")
            print("="*60)
            
            all_predictions = {}
            
            analyses = [
                ("1. Analysis Frequency", self.frequency_analysis_with_predictions),
                ("2. Analysis Digit", self.digit_analysis_with_predictions),
                ("3. Analysis Hot vs Cold Number", lambda: self.hot_cold_analysis_with_predictions(30)),
                ("4. Analysis of Even & Odd Numbers", self.even_odd_analysis_with_predictions),
                ("5. Analysis of The Sum of Digitst", self.digit_sum_analysis_with_predictions),
                ("6. Analysis Digit Repetition", self.digit_repetition_analysis_with_predictions),
                ("7. Analysis Pattern", self.pattern_analysis_with_predictions),
                ("8. Analysis of Prize Position", self.prize_position_analysis_with_predictions),
                ("9. Analysis Sliding Window", lambda: self.sliding_window_analysis_with_predictions(20)),
                ("10. Analysis Comprehensive Statistical", self.statistics_analysis_with_predictions),
                ("11. Analysis of the Rarest Numbers", self.new_numbers_analysis_with_predictions)
            ]
            
            for name, analysis_func in analyses:
                #print(f"\n▶️  {name}")
                #print("-" * 100)
                try:
                    predictions, _ = analysis_func()
                    all_predictions[name] = predictions
                    
                    print(f"\n   📋 Predictions: {', '.join(predictions)}")
                    input("\n   ⏸️  Press Enter to Continue...")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue
            
            # analysis Predictions Populer
            print("\n" + "="*60)
            print("📊 ANALYSIS PREDICTIONS POPULER")
            print("="*60)
            
            rekomendasi_akhir = self.predictions_populer_analysis(all_predictions)
            
            # Summary
            print("\n" + "="*60)
            print("📊 SUMMARY OF ALL PREDICTIONS")
            print("="*60)
            
            for name, predictions in all_predictions.items():
                print(f"\n{name}:")
                print(f"   {', '.join(predictions)}")
            
            # Find most common predictions
            print("\n" + "="*60)
            print("🎯 TOP 10 MOST POPULAR PREDICTIONS")
            print("="*60)
            
            all_preds_flat = []
            for predictions in all_predictions.values():
                all_preds_flat.extend(predictions)
            
            if all_preds_flat:
                pred_counter = Counter(all_preds_flat)
                
                print("\n🏆 PREDICTIONS WITH THE HIGHEST FREQUENCY:")
                for pred, count in pred_counter.most_common(10):
                    confidence = "🔥" * min(count, 3)
                    print(f"   • {pred}: {count} analysis {confidence}")
                
                # Tampilkan rekomendasi akhir
                if rekomendasi_akhir:
                    print(f"\n💡 FINAL RECOMMENDATION BASED ON CONSENSUS:")
                    for i, pred in enumerate(rekomendasi_akhir[:5], 1):
                        count = pred_counter.get(pred, 0)
                        print(f"   {i}. {pred} (appear in {count} analysis)")
            
            return all_predictions, rekomendasi_akhir


def main():
    print("\n" + "="*60)
    print("🎯 MALAYSIA - 4D [TOTO] SPORTSTOTO / [88] SABAH 88 - ALL ANALYSES WITH PREDICTIONS")
    print("="*60)
    
    analyzer = TOTO4DAnalyzer()
    
    while True:
        print("\n" + "="*60)
        print("🏠 MAIN MENU")
        print("="*60)
        
        print("\nPlease select :")
        print("  1.  📂 LOAD DATA (REQUIRED)")
        print("  2.  🚀 RUN ALL Analysis + Predictions")
        print("  3.  📊 Analysis Frequency + 5 Predictions")
        print("  4.  📊 Analysis Digit + 5 Predictions")
        print("  5.  📊 Analysis Hot vs Cold Number + 5 Predictions")
        print("  6.  📊 Analysis of Even & Odd Numbers + 5 Predictions")
        print("  7.  📊 Analysis of The Sum of Digitst + 5 Predictions")
        print("  8.  📊 Analysis Digit Repetition + 5 Predictions")
        print("  9.  📊 Analysis Pattern + 5 Predictions")
        print("  10. 📊 Analysis of Prize Position + 5 Predictions")
        print("  11. 📊 Analysis Sliding Window + 5 Predictions")
        print("  12. 📊 Analysis Comprehensive Statistical + 5 Predictions")
        print("  13. 📊 Analysis of the Rarest Numbers + 5 Predictions")
        print("  14. 🚀 Run Final Analysis Predictions Populer")
        print("  15. 💾 Export FULL Complete Reports")
        print("  16. ❌ EXIT")
        
        choice = input("\nChoice (1-16) : ").strip()
        
        if choice == '1':
            file_path = input("Path File Data : ").strip()
            if file_path and os.path.exists(file_path):
                success = analyzer.load_data_large(file_path)
                if success:
                    print(f"✅ Data Loaded Successfully: {len(analyzer.data):,} lines")
                else:
                    print("❌ Failed to Load Data")
            else:
                print("❌ File Not Found!")
        
        elif choice == '2':
            if analyzer.data is not None:
                analyzer.run_all_analyses_with_predictions(export_mode=False)
            else:
                print("❌ Please Load The Data First!")
        
        elif choice in ['3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']:
            if analyzer.data is not None:
                analysis_map = {
                    '3': ("Analysis Frequency", analyzer.frequency_analysis_with_predictions),
                    '4': ("Analysis Digit", analyzer.digit_analysis_with_predictions),
                    '5': ("Analysis Hot vs Cold Number", lambda: analyzer.hot_cold_analysis_with_predictions(30)),
                    '6': ("Analysis of Even & Odd Numbers", analyzer.even_odd_analysis_with_predictions),
                    '7': ("Analysis of The Sum of Digitst", analyzer.digit_sum_analysis_with_predictions),
                    '8': ("Analysis Digit Repetition", analyzer.digit_repetition_analysis_with_predictions),
                    '9': ("Analysis Pattern", analyzer.pattern_analysis_with_predictions),
                    '10': ("Analysis of Prize Position", analyzer.prize_position_analysis_with_predictions),
                    '11': ("Analysis Sliding Window", lambda: analyzer.sliding_window_analysis_with_predictions(20)),
                    '12': ("Analysis Comprehensive Statistical", analyzer.statistics_analysis_with_predictions),
                    '13': ("Analysis of the Rarest Numbers", analyzer.new_numbers_analysis_with_predictions)
                }

                if choice in analysis_map:
                    name, func = analysis_map[choice]
                    print(f"\n▶️  {name}")
                    print("="*60)
                    func()
            else:
                print("❌ Please Load The Data First!")
        
        elif choice == '14':
            if analyzer.data is not None:
                print("\n" + "="*60)
                print("📊 ANALYSIS PREDICTIONS POPULER")
                print("="*60)
                
                # Jalankan semua analysis terlebih dahulu
                print("\n⏳ Run all Analysis to gather predictions...")
                all_predictions, _ = analyzer.run_all_analyses_with_predictions(export_mode=False)
                
                # Tampilkan analysis predictions populer
                input("\n⏸️  Press Enter to see Popular Predictions Analysis...")
                analyzer.predictions_populer_analysis(all_predictions)
            else:
                print("❌ Please Load The Data First!")
        
        elif choice == '15':
            if analyzer.data is not None:
                filename = input("Name File Output (default: predictions_report.txt): ").strip()
                if not filename:
                    filename = "predictions_report.txt"
                
                print(f"⏳ Make a Report '{filename}'...")
                
                # Save original stdout
                original_stdout = sys.stdout
                
                try:
                    # Create output file
                    with open(filename, 'w', encoding='utf-8') as f:
                        # Redirect stdout to file
                        sys.stdout = f
                        
                        print("="*60)
                        print("REPORT MALAYSIA - 4D [TOTO] SPORTSTOTO / [88] SABAH 88 ")
                        print("="*60)
                        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"Data: {len(analyzer.data):,} vote")
                        print(f"Total Number: {len(analyzer.all_numbers_flat):,}")
                        print("="*60)
                        
                        # Run analyses in export mode
                        analyzer.run_all_analyses_with_predictions(export_mode=True)
                    
                    # Restore stdout
                    sys.stdout = original_stdout
                    
                    print(f"✅ Report Successfully exported to: {filename}")
                    print(f"📄 Size file: {os.path.getsize(filename):,} bytes")
                    
                except Exception as e:
                    # Make sure stdout is restored
                    sys.stdout = original_stdout
                    print(f"❌ Failed to Export Report : {e}")
                    
            else:
                print("❌ Please Load The Data First!")
        
        elif choice == '16':
            print("\n" + "="*60)
            print("🎯 GOOD LUCK HE-HE-HE-HE 👋")
            print("="*60)
            break
        
        else:
            print("❌ Invalid Selection!")
        
        if choice != '16':
            input("\n⏸️  Press Enter to Return...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Program Stopped!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
