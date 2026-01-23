import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import itertools
from typing import List, Dict, Tuple, Set
import warnings
import math
import sys
warnings.filterwarnings('ignore')

class TOTOPredictor40Analisis:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        self.numbers_4d = None
        self.draw_dates = None
        self.digits_counter = None
        self.all_pattern_stats = {}
        self.load_data()
    
    def load_data(self):
        """Load and preprocess data"""
        try:
            with open(self.file_path, 'r') as f:
                lines = f.readlines()
            
            all_numbers = []
            draw_dates = []
            dates_list = []
            
            for line in lines:
                if line.startswith('Draw_Date') or line.strip() == '':
                    continue
                    
                parts = line.strip().split(',')
                if len(parts) >= 24:
                    date_str = parts[0]
                    draw_dates.append(date_str)
                    dates_list.append(datetime.strptime(date_str, '%Y-%m-%d'))
                    
                    for num in parts[1:24]:
                        if len(num) == 4 and num.isdigit():
                            all_numbers.append(num)
            
            self.data = pd.DataFrame({
                'date': draw_dates * 23,
                'number': all_numbers,
                'draw_date': draw_dates * 23
            })
            
            self.numbers_4d = all_numbers
            self.draw_dates = dates_list
            self.digits_counter = Counter(''.join(all_numbers))
            
            print(f"✓ Data loaded: {len(draw_dates)} draws, {len(all_numbers)} numbers")
            
        except Exception as e:
            print(f"Error: {e}")

    # ==================== 40 ANALISIS CORAK ====================
    
    def analyze_1_Sequential_Up(self, num: str) -> bool:
        """1. Sequential Naik"""
        digits = [int(d) for d in num]
        for i in range(3):
            expected = (digits[i] + 1) % 10
            if digits[i + 1] != expected:
                return False
        return True
    
    def analyze_2_Sequential_Down(self, num: str) -> bool:
        """2. Sequential Turun"""
        digits = [int(d) for d in num]
        for i in range(3):
            expected = (digits[i] - 1) % 10
            if digits[i + 1] != expected:
                return False
        return True
    
    def analyze_3_palindrome(self, num: str) -> bool:
        """3. Palindrome"""
        return num == num[::-1]
    
    def analyze_4_mirror_abba(self, num: str) -> bool:
        """4. Mirror (ABBA)"""
        return num[0] == num[3] and num[1] == num[2]
    
    def analyze_5_repeat_aabb(self, num: str) -> bool:
        """5. Repeat Pattern (AABB)"""
        return num[0] == num[1] and num[2] == num[3] and num[0] != num[2]
    
    def analyze_6_alternating_abab(self, num: str) -> bool:
        """6. Alternating (ABAB)"""
        return num[0] == num[2] and num[1] == num[3] and num[0] != num[1]
    
    def analyze_7_All_Even(self, num: str) -> bool:
        """7. Semua Genap"""
        return all(int(d) % 2 == 0 for d in num)
    
    def analyze_8_All_Odd(self, num: str) -> bool:
        """8. Semua Ganjil"""
        return all(int(d) % 2 == 1 for d in num)
    
    def analyze_9_Mixed_Even_Odd(self, num: str) -> bool:
        """9. Campuran Genap-Ganjil"""
        genap = sum(1 for d in num if int(d) % 2 == 0)
        return 1 <= genap <= 3
    
    def analyze_10_Small_0_4(self, num: str) -> bool:
        """10. Kecil (0-4)"""
        return all(0 <= int(d) <= 4 for d in num)
    
    def analyze_11_Big_5_9(self, num: str) -> bool:
        """11. Besar (5-9)"""
        return all(5 <= int(d) <= 9 for d in num)
    
    def analyze_12_Big_Small_Mix(self, num: str) -> bool:
        """12. Campuran Kecil-Besar"""
        kecil = sum(1 for d in num if 0 <= int(d) <= 4)
        return 1 <= kecil <= 3
    
    def analyze_13_aritmatika(self, num: str) -> Tuple[bool, int]:
        """13. Aritmatika (Difference tetap)"""
        digits = [int(d) for d in num]
        diffs = [digits[i+1] - digits[i] for i in range(3)]
        if len(set(diffs)) == 1 and diffs[0] != 0:
            return True, diffs[0]
        return False, 0
    
    def analyze_14_geometri(self, num: str) -> Tuple[bool, float]:
        """14. Geometri"""
        digits = [int(d) for d in num]
        if 0 in digits:
            return False, 0
        
        ratios = []
        for i in range(3):
            if digits[i] == 0:
                return False, 0
            ratios.append(digits[i+1] / digits[i])
        
        if abs(max(ratios) - min(ratios)) < 0.001:
            return True, ratios[0]
        return False, 0
    
    def analyze_15_fibonacci_like(self, num: str) -> bool:
        """15. Fibonacci-like"""
        digits = [int(d) for d in num]
        return (digits[2] == (digits[0] + digits[1]) % 10 and 
                digits[3] == (digits[1] + digits[2]) % 10)
    
    def analyze_16_birthday_pattern(self, num: str) -> str:
        """16. Birthday Pattern"""
        # DDMM
        dd = int(num[:2])
        mm = int(num[2:])
        if 1 <= dd <= 31 and 1 <= mm <= 12:
            return "DDMM"
        
        # MMDD
        if 1 <= int(num[2:]) <= 31 and 1 <= int(num[:2]) <= 12:
            return "MMDD"
        
        # YYYY (last 4 digits of year)
        year = int(num)
        if 1900 <= year <= 2100:
            return "YYYY"
        
        return ""
    
    def analyze_17_mountain(self, num: str) -> bool:
        """17. Mountain - naik lalu turun"""
        digits = [int(d) for d in num]
        return (digits[0] < digits[1] and 
                digits[1] > digits[2] and 
                digits[2] > digits[3])
    
    def analyze_18_valley(self, num: str) -> bool:
        """18. Valley - turun lalu naik"""
        digits = [int(d) for d in num]
        return (digits[0] > digits[1] and 
                digits[1] < digits[2] and 
                digits[2] < digits[3])
    
    def analyze_19_plateau(self, num: str) -> bool:
        """19. Plateau - datar di tengah"""
        return num[1] == num[2] and num[0] != num[1] and num[2] != num[3]
    
    def analyze_20_cliff(self, num: str) -> bool:
        """20. Cliff - lompatan besar"""
        digits = [int(d) for d in num]
        return any(abs(digits[i+1] - digits[i]) >= 5 for i in range(3))
    
    def analyze_21_double_pair(self, num: str) -> bool:
        """21. Double Pair - AABB"""
        return len(set(num)) == 2 and num[0] == num[1] and num[2] == num[3]
    
    def analyze_22_triple(self, num: str) -> bool:
        """22. Triple - AAAB"""
        counts = Counter(num)
        return 3 in counts.values() and len(counts) == 2
    
    def analyze_23_quad(self, num: str) -> bool:
        """23. Quad - AAAA"""
        return len(set(num)) == 1
    
    def analyze_24_all_different(self, num: str) -> bool:
        """24. All Different - ABCD"""
        return len(set(num)) == 4
    
    def analyze_25_first_last_same(self, num: str) -> bool:
        """25. First-Last Same"""
        return num[0] == num[3]
    
    def analyze_26_middle_same(self, num: str) -> bool:
        """26. Middle Same"""
        return num[1] == num[2]
    
    def analyze_27_bookend(self, num: str) -> bool:
        """27. Bookend - ABBA"""
        return num[0] == num[3] and num[1] == num[2] and num[0] != num[1]
    
    def analyze_28_Small_Total(self, num: str) -> bool:
        """28. Total Kecil (0-9)"""
        total = sum(int(d) for d in num)
        return 0 <= total <= 9
    
    def analyze_29_Medium_Total(self, num: str) -> bool:
        """29. Total Sedang (10-18)"""
        total = sum(int(d) for d in num)
        return 10 <= total <= 18
    
    def analyze_30_Large_Total(self, num: str) -> bool:
        """30. Total Besar (19-27)"""
        total = sum(int(d) for d in num)
        return 19 <= total <= 27
    
    def analyze_31_Extreme_Total(self, num: str) -> bool:
        """31. Total Ekstrem (28-36)"""
        total = sum(int(d) for d in num)
        return 28 <= total <= 36
    
    def get_hot_cold_digits(self) -> Dict:
        """Get hot and cold digits"""
        if not self.digits_counter:
            return {'hot': [], 'cold': [], 'all': {}}
        
        avg_freq = sum(self.digits_counter.values()) / 10
        digits_freq = {str(i): self.digits_counter.get(str(i), 0) for i in range(10)}
        
        hot = [d for d, f in digits_freq.items() if f > avg_freq * 1.2]
        cold = [d for d, f in digits_freq.items() if f < avg_freq * 0.8]
        
        return {'hot': hot, 'cold': cold, 'all': digits_freq}
    
    def analyze_32_hot_digits(self, num: str) -> bool:
        """32. Hot Digits"""
        hot_digits = self.get_hot_cold_digits()['hot']
        return sum(1 for d in num if d in hot_digits) >= 3
    
    def analyze_33_cold_digits(self, num: str) -> bool:
        """33. Cold Digits"""
        cold_digits = self.get_hot_cold_digits()['cold']
        return sum(1 for d in num if d in cold_digits) >= 3
    
    def analyze_34_balanced_digits(self, num: str) -> bool:
        """34. Balanced Digits"""
        hc = self.get_hot_cold_digits()
        hot_count = sum(1 for d in num if d in hc['hot'])
        cold_count = sum(1 for d in num if d in hc['cold'])
        return 1 <= hot_count <= 2 and 1 <= cold_count <= 2
    
    def analyze_35_lucky_number(self, num: str) -> bool:
        """35. Lucky Number"""
        lucky_numbers = {'1688', '1314', '8888', '9999', '1111', '2222', 
                        '3333', '4444', '5555', '6666', '7777', '5200', 
                        '3344', '1133', '2233', '1122', '1221', '1331'}
        return num in lucky_numbers or any(lucky in num for lucky in ['168', '131', '888', '999'])
    
    def analyze_36_historical_pattern(self, num: str) -> bool:
        """36. Historical Pattern"""
        if len(self.numbers_4d) < 10:
            return False
        
        recent = self.numbers_4d[-10:]
        for recent_num in recent:
            same_digits = sum(1 for a, b in zip(num, recent_num) if a == b)
            if same_digits >= 3:
                return True
        return False
    
    def analyze_37_seasonal_pattern(self, num: str, date_str: str = None) -> bool:
        """37. Seasonal Pattern"""
        try:
            if date_str:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                month = date_obj.month
            else:
                month = datetime.now().month
            
            # Seasonal patterns
            seasonal_numbers = {
                1: ['1111', '2222', '0101', '0110', '1001'],
                2: ['0202', '1414', '2323', '1314', '0214'],
                3: ['0303', '0312', '1221', '1324', '2413'],
                4: ['0404', '0415', '1524', '0420', '2004'],
                5: ['0505', '0515', '1520', '0525', '2505'],
                6: ['0606', '0618', '1824', '0630', '3006'],
                7: ['0707', '0714', '1421', '0728', '2807'],
                8: ['0808', '0816', '1624', '0831', '3108'],
                9: ['0909', '0918', '1827', '0930', '3009'],
                10: ['1010', '1020', '2030', '1031', '3110'],
                11: ['1111', '1122', '2233', '1130', '3011'],
                12: ['1212', '1225', '2512', '1231', '3112']
            }
            
            if month in seasonal_numbers:
                return num in seasonal_numbers[month]
            
        except:
            pass
        return False
    
    def analyze_38_date_based(self, num: str, date_str: str = None) -> bool:
        """38. Date-based"""
        try:
            if date_str:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            else:
                date_obj = datetime.now()
                
            day = date_obj.day
            month = date_obj.month
            year_last2 = date_obj.year % 100
            
            date_patterns = [
                f"{day:02d}{month:02d}",
                f"{month:02d}{day:02d}",
                f"{day:02d}{year_last2:02d}",
                f"{month:02d}{year_last2:02d}",
                f"{year_last2:02d}{month:02d}",
                f"{year_last2:02d}{day:02d}"
            ]
            
            return any(pattern in num for pattern in date_patterns)
        except:
            return False
    
    def analyze_39_not_appeared(self, num: str) -> bool:
        """39. Nombor yang belum keluar"""
        all_numbers_set = set(self.numbers_4d)
        return num not in all_numbers_set
    
    def analyze_40_special_combination(self, num: str) -> bool:
        """40. Special Combination - mix of all patterns"""
        # Check if number has multiple special patterns
        pattern_count = 0
        
        if self.analyze_3_palindrome(num):
            pattern_count += 1
        if self.analyze_4_mirror_abba(num):
            pattern_count += 1
        if self.analyze_5_repeat_aabb(num):
            pattern_count += 1
        if self.analyze_6_alternating_abab(num):
            pattern_count += 1
        if self.analyze_13_aritmatika(num)[0]:
            pattern_count += 1
        if self.analyze_15_fibonacci_like(num):
            pattern_count += 1
        
        return pattern_count >= 2
    
    # ==================== ANALISIS SEMUA CORAK ====================
    
    def analyze_all_patterns_for_number(self, num: str, date_str: str = None) -> Dict:
        """Analyze all 40 patterns for a single number"""
        patterns = {}
        
        patterns['1_Sequential_Up'] = self.analyze_1_Sequential_Up(num)
        patterns['2_Sequential_Down'] = self.analyze_2_Sequential_Down(num)
        patterns['3_Palindrome'] = self.analyze_3_palindrome(num)
        patterns['4_Mirror_ABBA'] = self.analyze_4_mirror_abba(num)
        patterns['5_Repeat_AABB'] = self.analyze_5_repeat_aabb(num)
        patterns['6_Alternating_ABAB'] = self.analyze_6_alternating_abab(num)
        patterns['7_All_Even'] = self.analyze_7_All_Even(num)
        patterns['8_All_Odd'] = self.analyze_8_All_Odd(num)
        patterns['9_Mixed_Even_Odd'] = self.analyze_9_Mixed_Even_Odd(num)
        patterns['10_Small_0_4'] = self.analyze_10_Small_0_4(num)
        patterns['11_Big_5_9'] = self.analyze_11_Big_5_9(num)
        patterns['12_Big_Small_Mix'] = self.analyze_12_Big_Small_Mix(num)
        
        arit_result = self.analyze_13_aritmatika(num)
        patterns['13_Aritmatika'] = arit_result[0]
        patterns['13_Aritmatika_Difference'] = arit_result[1] if arit_result[0] else 0
        
        geo_result = self.analyze_14_geometri(num)
        patterns['14_Geometri'] = geo_result[0]
        patterns['14_Geometri_Ratio'] = geo_result[1] if geo_result[0] else 0
        
        patterns['15_Fibonacci_Like'] = self.analyze_15_fibonacci_like(num)
        patterns['16_Birthday_Pattern'] = self.analyze_16_birthday_pattern(num)
        patterns['17_Mountain'] = self.analyze_17_mountain(num)
        patterns['18_Valley'] = self.analyze_18_valley(num)
        patterns['19_Plateau'] = self.analyze_19_plateau(num)
        patterns['20_Cliff'] = self.analyze_20_cliff(num)
        patterns['21_Double_Pair'] = self.analyze_21_double_pair(num)
        patterns['22_Triple'] = self.analyze_22_triple(num)
        patterns['23_Quad'] = self.analyze_23_quad(num)
        patterns['24_All_Different'] = self.analyze_24_all_different(num)
        patterns['25_First_Last_Same'] = self.analyze_25_first_last_same(num)
        patterns['26_Middle_Same'] = self.analyze_26_middle_same(num)
        patterns['27_Bookend'] = self.analyze_27_bookend(num)
        patterns['28_Small_Total'] = self.analyze_28_Small_Total(num)
        patterns['29_Medium_Total'] = self.analyze_29_Medium_Total(num)
        patterns['30_Large_Total'] = self.analyze_30_Large_Total(num)
        patterns['31_Extreme_Total'] = self.analyze_31_Extreme_Total(num)
        patterns['32_Hot_Digits'] = self.analyze_32_hot_digits(num)
        patterns['33_Cold_Digits'] = self.analyze_33_cold_digits(num)
        patterns['34_Balanced_Digits'] = self.analyze_34_balanced_digits(num)
        patterns['35_Lucky_Number'] = self.analyze_35_lucky_number(num)
        patterns['36_Historical_Pattern'] = self.analyze_36_historical_pattern(num)
        patterns['37_Seasonal_Pattern'] = self.analyze_37_seasonal_pattern(num, date_str)
        patterns['38_Date_Based'] = self.analyze_38_date_based(num, date_str)
        patterns['39_Not_Appeared'] = self.analyze_39_not_appeared(num)
        patterns['40_Special_Combination'] = self.analyze_40_special_combination(num)
        
        return patterns
    
    # ==================== GENERATE PREDICTIONS ====================
    
    def generate_predictions_for_pattern(self, pattern_id: int, count: int = 2) -> List[str]:
        """Generate 2 predictions for each pattern"""
        predictions = []
        
        if pattern_id == 1:  # Sequential Naik
            for start in range(10):
                num = ''.join(str((start + i) % 10) for i in range(4))
                predictions.append(num)
            predictions = predictions[:10]
        
        elif pattern_id == 2:  # Sequential Turun
            for start in range(10):
                num = ''.join(str((start - i) % 10) for i in range(4))
                predictions.append(num)
            predictions = predictions[:10]
        
        elif pattern_id == 3:  # Palindrome
            for i in range(10):
                for j in range(10):
                    num = f"{i}{j}{j}{i}"
                    predictions.append(num)
        
        elif pattern_id == 4:  # Mirror ABBA
            for i in range(10):
                for j in range(10):
                    if i != j:
                        num = f"{i}{j}{j}{i}"
                        predictions.append(num)
        
        elif pattern_id == 5:  # Repeat AABB
            for i in range(10):
                for j in range(10):
                    if i != j:
                        num = f"{i}{i}{j}{j}"
                        predictions.append(num)
        
        elif pattern_id == 6:  # Alternating ABAB
            for i in range(10):
                for j in range(10):
                    if i != j:
                        num = f"{i}{j}{i}{j}"
                        predictions.append(num)
        
        elif pattern_id == 7:  # Semua Genap
            evens = [str(i) for i in range(0, 10, 2)]
            for _ in range(20):
                num = ''.join(np.random.choice(evens, 4))
                predictions.append(num)
        
        elif pattern_id == 8:  # Semua Ganjil
            odds = [str(i) for i in range(1, 10, 2)]
            for _ in range(20):
                num = ''.join(np.random.choice(odds, 4))
                predictions.append(num)
        
        elif pattern_id == 9:  # Campuran Genap-Ganjil
            for _ in range(50):
                num = ''.join(str(np.random.randint(0, 10)) for _ in range(4))
                if self.analyze_9_Mixed_Even_Odd(num):
                    predictions.append(num)
        
        elif pattern_id == 10:  # Kecil 0-4
            small = [str(i) for i in range(5)]
            for _ in range(20):
                num = ''.join(np.random.choice(small, 4))
                predictions.append(num)
        
        elif pattern_id == 11:  # Besar 5-9
            large = [str(i) for i in range(5, 10)]
            for _ in range(20):
                num = ''.join(np.random.choice(large, 4))
                predictions.append(num)
        
        elif pattern_id == 12:  # Campuran Kecil-Besar
            small = [str(i) for i in range(5)]
            large = [str(i) for i in range(5, 10)]
            for _ in range(50):
                num = ''.join(np.random.choice(small, 2)) + ''.join(np.random.choice(large, 2))
                predictions.append(num)
        
        elif pattern_id == 13:  # Aritmatika
            for diff in [1, 2, 3, -1, -2, -3]:
                for start in range(10):
                    digits = [(start + diff * i) % 10 for i in range(4)]
                    num = ''.join(str(d) for d in digits)
                    predictions.append(num)
        
        elif pattern_id == 14:  # Geometri
            for ratio in [2, 3, 0.5]:
                for start in range(1, 10):
                    digits = [start]
                    for i in range(3):
                        next_val = int(digits[-1] * ratio)
                        digits.append(next_val % 10)
                    num = ''.join(str(d % 10) for d in digits)
                    predictions.append(num)
        
        elif pattern_id == 15:  # Fibonacci-like
            for a in range(10):
                for b in range(10):
                    c = (a + b) % 10
                    d = (b + c) % 10
                    num = f"{a}{b}{c}{d}"
                    predictions.append(num)
        
        elif pattern_id == 16:  # Birthday Pattern
            for day in range(1, 32):
                for month in range(1, 13):
                    num = f"{day:02d}{month:02d}"
                    predictions.append(num)
            predictions = predictions[:100]
        
        elif pattern_id == 17:  # Mountain
            for _ in range(50):
                a = np.random.randint(0, 7)
                b = np.random.randint(a+1, 10)
                c = np.random.randint(0, b)
                d = np.random.randint(0, c)
                if b > c > d:
                    num = f"{a}{b}{c}{d}"
                    predictions.append(num)
        
        elif pattern_id == 18:  # Valley
            for _ in range(50):
                a = np.random.randint(3, 10)
                b = np.random.randint(0, a)
                c = np.random.randint(b+1, 10)
                d = np.random.randint(c+1, 10)
                if a > b and c > b and d > c:
                    num = f"{a}{b}{c}{d}"
                    predictions.append(num)
        
        elif pattern_id == 19:  # Plateau
            for i in range(10):
                for j in range(10):
                    if i != j:
                        for k in range(10):
                            if k != j:
                                num = f"{i}{j}{j}{k}"
                                predictions.append(num)
        
        elif pattern_id == 20:  # Cliff
            for _ in range(50):
                digits = [np.random.randint(0, 10) for _ in range(4)]
                if any(abs(digits[i+1] - digits[i]) >= 5 for i in range(3)):
                    num = ''.join(str(d) for d in digits)
                    predictions.append(num)
        
        elif pattern_id == 21:  # Double Pair
            for i in range(10):
                for j in range(10):
                    if i != j:
                        num = f"{i}{i}{j}{j}"
                        predictions.append(num)
        
        elif pattern_id == 22:  # Triple
            for i in range(10):
                for j in range(10):
                    if i != j:
                        predictions.append(f"{i}{i}{i}{j}")
                        predictions.append(f"{j}{i}{i}{i}")
                        predictions.append(f"{i}{j}{j}{j}")
        
        elif pattern_id == 23:  # Quad
            for i in range(10):
                num = f"{i}{i}{i}{i}"
                predictions.append(num)
        
        elif pattern_id == 24:  # All Different
            for _ in range(100):
                digits = np.random.choice(range(10), 4, replace=False)
                num = ''.join(str(d) for d in digits)
                predictions.append(num)
        
        elif pattern_id == 25:  # First-Last Same
            for _ in range(50):
                a = str(np.random.randint(0, 10))
                b = str(np.random.randint(0, 10))
                c = str(np.random.randint(0, 10))
                num = f"{a}{b}{c}{a}"
                predictions.append(num)
        
        elif pattern_id == 26:  # Middle Same
            for _ in range(50):
                a = str(np.random.randint(0, 10))
                b = str(np.random.randint(0, 10))
                c = str(np.random.randint(0, 10))
                num = f"{a}{b}{b}{c}"
                predictions.append(num)
        
        elif pattern_id == 27:  # Bookend
            for i in range(10):
                for j in range(10):
                    if i != j:
                        num = f"{i}{j}{j}{i}"
                        predictions.append(num)
        
        elif pattern_id == 28:  # Total Kecil (0-9)
            for _ in range(100):
                while True:
                    num = f"{np.random.randint(0, 3)}{np.random.randint(0, 3)}"
                    num += f"{np.random.randint(0, 3)}{np.random.randint(0, 3)}"
                    total = sum(int(d) for d in num)
                    if 0 <= total <= 9:
                        predictions.append(num)
                        break
        
        elif pattern_id == 29:  # Total Sedang (10-18)
            for _ in range(100):
                while True:
                    num = f"{np.random.randint(0, 10)}{np.random.randint(0, 10)}"
                    num += f"{np.random.randint(0, 10)}{np.random.randint(0, 10)}"
                    total = sum(int(d) for d in num)
                    if 10 <= total <= 18:
                        predictions.append(num)
                        break
        
        elif pattern_id == 30:  # Total Besar (19-27)
            for _ in range(100):
                while True:
                    num = f"{np.random.randint(5, 10)}{np.random.randint(5, 10)}"
                    num += f"{np.random.randint(0, 10)}{np.random.randint(0, 10)}"
                    total = sum(int(d) for d in num)
                    if 19 <= total <= 27:
                        predictions.append(num)
                        break
        
        elif pattern_id == 31:  # Total Ekstrem (28-36)
            for _ in range(100):
                while True:
                    num = f"{np.random.randint(7, 10)}{np.random.randint(7, 10)}"
                    num += f"{np.random.randint(7, 10)}{np.random.randint(7, 10)}"
                    total = sum(int(d) for d in num)
                    if 28 <= total <= 36:
                        predictions.append(num)
                        break
        
        elif pattern_id == 32:  # Hot Digits
            hot = self.get_hot_cold_digits()['hot']
            if hot:
                for _ in range(50):
                    num = ''.join(np.random.choice(hot, 4))
                    predictions.append(num)
            else:
                predictions = ['1111', '2222', '3333']
        
        elif pattern_id == 33:  # Cold Digits
            cold = self.get_hot_cold_digits()['cold']
            if cold:
                for _ in range(50):
                    num = ''.join(np.random.choice(cold, 4))
                    predictions.append(num)
            else:
                predictions = ['4444', '5555', '6666']
        
        elif pattern_id == 34:  # Balanced Digits
            hc = self.get_hot_cold_digits()
            hot = hc['hot']
            cold = hc['cold']
            if hot and cold:
                for _ in range(50):
                    hot_part = ''.join(np.random.choice(hot, 2))
                    cold_part = ''.join(np.random.choice(cold, 2))
                    chars = list(hot_part + cold_part)
                    np.random.shuffle(chars)
                    num = ''.join(chars)
                    predictions.append(num)
            else:
                predictions = ['1234', '5678', '9012']
        
        elif pattern_id == 35:  # Lucky Number
            lucky_numbers = ['1688', '1314', '8888', '9999', '1111', '2222', '3333', 
                            '4444', '5555', '6666', '7777', '1234', '4321', '1122',
                            '2233', '3344', '4455', '5566', '6677', '7788', '8899',
                            '9900', '0088', '1188', '2288', '3388', '4488', '5588']
            predictions = lucky_numbers
        
        elif pattern_id == 36:  # Historical Pattern
            if len(self.numbers_4d) >= 5:
                recent = self.numbers_4d[-5:]
                for num in recent:
                    for pos in range(4):
                        for change in [-1, 1]:
                            new_digit = (int(num[pos]) + change) % 10
                            new_num = list(num)
                            new_num[pos] = str(new_digit)
                            predictions.append(''.join(new_num))
            else:
                predictions = ['1234', '5678', '9876']
        
        elif pattern_id == 37:  # Seasonal Pattern
            current_month = datetime.now().month
            seasonal = {
                1: ['1111', '0101', '0110', '1001', '1100'],
                2: ['0202', '1414', '2323', '1314', '0214'],
                3: ['0303', '0312', '1221', '1324', '2413'],
                4: ['0404', '0415', '1524', '0420', '2004'],
                5: ['0505', '0515', '1520', '0525', '2505'],
                6: ['0606', '0618', '1824', '0630', '3006'],
                7: ['0707', '0714', '1421', '0728', '2807'],
                8: ['0808', '0816', '1624', '0831', '3108'],
                9: ['0909', '0918', '1827', '0930', '3009'],
                10: ['1010', '1020', '2030', '1031', '3110'],
                11: ['1111', '1122', '2233', '1130', '3011'],
                12: ['1212', '1225', '2512', '1231', '3112']
            }
            if current_month in seasonal:
                predictions = seasonal[current_month]
            else:
                predictions = [f"{current_month:02d}{i:02d}" for i in range(10, 20)]
        
        elif pattern_id == 38:  # Date-based
            today = datetime.now()
            predictions = [
                f"{today.day:02d}{today.month:02d}",
                f"{today.month:02d}{today.day:02d}",
                f"{today.year % 100:02d}{today.month:02d}",
                f"{today.month:02d}{today.year % 100:02d}",
                f"{today.day:02d}{today.year % 100:02d}",
                f"{today.year % 100:02d}{today.day:02d}"
            ]
        
        elif pattern_id == 39:  # Not Appeared
            all_numbers_set = set(self.numbers_4d)
            not_appeared = []
            for i in range(10000):
                num = f"{i:04d}"
                if num not in all_numbers_set:
                    not_appeared.append(num)
                    if len(not_appeared) >= 100:
                        break
            predictions = not_appeared
        
        elif pattern_id == 40:  # Special Combination
            # Generate numbers with multiple patterns
            special_numbers = []
            
            # Check existing numbers for special combinations
            for num in set(self.numbers_4d):
                if self.analyze_40_special_combination(num):
                    special_numbers.append(num)
            
            if not special_numbers:
                # Generate new special combinations
                combinations = [
                    '1221',  # Palindrome + Mirror
                    '1331',  # Palindrome + Mirror
                    '1441',  # Palindrome + Mirror
                    '2332',  # Palindrome + Mirror
                    '3443',  # Palindrome + Mirror
                    '1112',  # Triple + Historical
                    '2223',  # Triple + Historical
                    '3334',  # Triple + Historical
                    '1122',  # Double Pair + Repeat
                    '2233',  # Double Pair + Repeat
                    '3344',  # Double Pair + Repeat
                    '1212',  # Alternating + Pattern
                    '1313',  # Alternating + Pattern
                    '1414',  # Alternating + Pattern
                    '1232',  # Mountain + Pattern
                    '1343',  # Mountain + Pattern
                    '1454',  # Mountain + Pattern
                    '4323',  # Valley + Pattern
                    '5434',  # Valley + Pattern
                    '6545',  # Valley + Pattern
                ]
                predictions = combinations
            else:
                predictions = special_numbers
        
        # Remove duplicates and limit to requested count
        predictions = list(set(predictions))
        np.random.shuffle(predictions)
        return predictions[:count]
    
    def generate_all_predictions(self) -> Dict[str, List[str]]:
        """Generate 2 predictions for each of the 40 patterns"""
        print("\n" + "="*80)
        print("TOTO 4D MALAYSIA - 40 PATTERN ANALYSIS WITH 2 PREDICTIONS EACH ANALYSIS")
        print("="*80)
        
        all_predictions = {}
        
        # Fixed pattern names list with exactly 40 items
        pattern_names = [
            "1.  Sequential_Up",
            "2.  Sequential_Down",
            "3.  Palindrome",
            "4.  Mirror_ABBA",
            "5.  Repeat_AABB",
            "6.  Alternating_ABAB",
            "7.  All_Even",
            "8.  All_Odd",
            "9.  Mixed_Even_Odd",
            "10. Small_0_4",
            "11. Big_5_9",
            "12. Big_Small_Mix",
            "13. Aritmatika",
            "14. Geometri",
            "15. Fibonacci_Like",
            "16. Birthday_Pattern",
            "17. Mountain",
            "18. Valley",
            "19. Plateau",
            "20. Cliff",
            "21. Double_Pair",
            "22. Triple",
            "23. Quad",
            "24. All_Different",
            "25. First_Last_Same",
            "26. Middle_Same",
            "27. Bookend",
            "28. Small_Total",
            "29. Medium_Total",
            "30. Large_Total",
            "31. Extreme_Total",
            "32. Hot_Digits",
            "33. Cold_Digits",
            "34. Balanced_Digits",
            "35. Lucky_Number",
            "36. Historical_Pattern",
            "37. Seasonal_Pattern",
            "38. Date_Based",
            "39. Not_Appeared",
            "40. Special_Combination"
        ]
        
        # Verify we have exactly 40 pattern names
        print(f"Total pattern names: {len(pattern_names)}")
        assert len(pattern_names) == 40, f"Expected 40 pattern names, got {len(pattern_names)}"
        
        for pattern_id in range(1, 41):
            try:
                predictions = self.generate_predictions_for_pattern(pattern_id, 2)
                pattern_name = pattern_names[pattern_id - 1]
                all_predictions[pattern_name] = predictions
                
                print(f"\n{pattern_name}:")
                for i, pred in enumerate(predictions, 1):
                    # Analyze this prediction
                    analysis = self.analyze_all_patterns_for_number(pred)
                    
                    # Get analysis info
                    info_parts = []
                    total = sum(int(d) for d in pred)
                    info_parts.append(f"Total: {total}")
                    
                    # Check if prediction matches its own pattern
                    pattern_match = False
                    pattern_desc = ""
                    
                    # Map pattern_id to analysis key
                    pattern_mapping = {
                        1: '1_Sequential_Up',
                        2: '2_Sequential_Down',
                        3: '3_Palindrome',
                        4: '4_Mirror_ABBA',
                        5: '5_Repeat_AABB',
                        6: '6_Alternating_ABAB',
                        7: '7_All_Even',
                        8: '8_All_Odd',
                        9: '9_Mixed_Even_Odd',
                        10: '10_Small_0_4',
                        11: '11_Big_5_9',
                        12: '12_Big_Small_Mix',
                        13: '13_Aritmatika',
                        14: '14_Geometri',
                        15: '15_Fibonacci_Like',
                        16: '16_Birthday_Pattern',
                        17: '17_Mountain',
                        18: '18_Valley',
                        19: '19_Plateau',
                        20: '20_Cliff',
                        21: '21_Double_Pair',
                        22: '22_Triple',
                        23: '23_Quad',
                        24: '24_All_Different',
                        25: '25_First_Last_Same',
                        26: '26_Middle_Same',
                        27: '27_Bookend',
                        28: '28_Small_Total',
                        29: '29_Medium_Total',
                        30: '30_Large_Total',
                        31: '31_Extreme_Total',
                        32: '32_Hot_Digits',
                        33: '33_Cold_Digits',
                        34: '34_Balanced_Digits',
                        35: '35_Lucky_Number',
                        36: '36_Historical_Pattern',
                        37: '37_Seasonal_Pattern',
                        38: '38_Date_Based',
                        39: '39_Not_Appeared',
                        40: '40_Special_Combination'
                    }
                    
                    if pattern_id in pattern_mapping:
                        analysis_key = pattern_mapping[pattern_id]
                        if analysis_key in analysis and analysis[analysis_key]:
                            pattern_match = True
                            pattern_desc = f"✓ {pattern_names[pattern_id-1].split('. ')[1]}"
                            
                            # Add special info for certain patterns
                            if pattern_id == 13 and analysis['13_Aritmatika_Difference']:
                                pattern_desc += f" (Difference {analysis['13_Aritmatika_Difference']})"
                            elif pattern_id == 14 and analysis['14_Geometri_Ratio']:
                                pattern_desc += f" (ratio {analysis['14_Geometri_Ratio']:.1f})"
                            elif pattern_id == 16 and analysis['16_Birthday_Pattern']:
                                pattern_desc += f" ({analysis['16_Birthday_Pattern']})"
                    
                    if not pattern_match:
                        pattern_desc = "Various patterns"
                    
                    # Hot/cold analysis
                    hc = self.get_hot_cold_digits()
                    hot_count = sum(1 for d in pred if d in hc['hot'])
                    cold_count = sum(1 for d in pred if d in hc['cold'])
                    
                    if hot_count > 0:
                        info_parts.append(f"Hot: {hot_count}")
                    if cold_count > 0:
                        info_parts.append(f"Cold: {cold_count}")
                    
                    info_str = " | ".join(info_parts)
                    print(f"  Prediction {i}: {pred} - {pattern_desc} [{info_str}]")
                    
            except Exception as e:
                print(f"Error generating predictions for pattern {pattern_id}: {e}")
                all_predictions[pattern_names[pattern_id - 1]] = ["0000", "1111"]
        
        # Summary statistics
        print("\n" + "="*80)
        print("STATISTIK ANALISIS:")
        print("-"*80)
        
        # Hot/Cold digits
        hc = self.get_hot_cold_digits()
        print(f"Digit HOT (often appear): {', '.join(hc['hot']) if hc['hot'] else 'None'}")
        print(f"Digit COLD (rarely appears): {', '.join(hc['cold']) if hc['cold'] else 'None'}")
        
        # Frequency of digits
        print(f"\nFrekuensi Digit (0-9):")
        for digit in range(10):
            freq = hc['all'].get(str(digit), 0)
            percentage = (freq / len(self.numbers_4d) * 100) if len(self.numbers_4d) > 0 else 0
            print(f"  {digit}: {freq} times ({percentage:.1f}%)")
        
        # Most common patterns in historical data
        print(f"\nMOST FREQUENT PATTERNS IN HISTORICAL DATA (first 100 numbers):")
        pattern_counts = defaultdict(int)
        sample_size = min(100, len(self.numbers_4d))
        for num in self.numbers_4d[:sample_size]:
            analysis = self.analyze_all_patterns_for_number(num)
            for key, value in analysis.items():
                if value and key not in ['13_Aritmatika_Difference', '14_Geometri_Ratio', '16_Birthday_Pattern']:
                    pattern_name = key.replace('_', ' ')
                    pattern_counts[pattern_name] += 1
        
        top_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        for pattern, count in top_patterns:
            percentage = (count / sample_size * 100)
            print(f"  {pattern}: {count} times ({percentage:.1f}%)")
        
        # Generate top 10 overall recommendations
        print("\n" + "="*80)
        print("10 MAIN RECOMMENDATIONS BASED ON ALL ANALYSI:")
        print("-"*80)
        
        all_recommended = []
        for preds in all_predictions.values():
            all_recommended.extend(preds)
        
        # Score each number
        scored_numbers = []
        for num in set(all_recommended):
            if num == "0000":  # Skip placeholder
                continue
                
            score = 0
            analysis = self.analyze_all_patterns_for_number(num)
            
            # Bonus for historical patterns
            if analysis['36_Historical_Pattern']:
                score += 3
            
            # Bonus for not appeared
            if analysis['39_Not_Appeared']:
                score += 2
            
            # Bonus for sum 10-18 (most common range)
            total = sum(int(d) for d in num)
            if 10 <= total <= 18:
                score += 2
            
            # Bonus for having 2-3 hot digits
            hc = self.get_hot_cold_digits()
            hot_count = sum(1 for d in num if d in hc['hot'])
            if 2 <= hot_count <= 3:
                score += hot_count
            
            # Bonus for all different digits
            if analysis['24_All_Different']:
                score += 1
            
            # Bonus for lucky numbers
            if analysis['35_Lucky_Number']:
                score += 2
            
            # Bonus for special combination
            if analysis['40_Special_Combination']:
                score += 3
            
            scored_numbers.append((num, score))
        
        # Sort by score
        scored_numbers.sort(key=lambda x: x[1], reverse=True)
        
        for i, (num, score) in enumerate(scored_numbers[:10], 1):
            patterns = []
            analysis = self.analyze_all_patterns_for_number(num)
            
            # Get top patterns
            pattern_list = []
            for key, value in analysis.items():
                if value and key not in ['13_Aritmatika_Difference', '14_Geometri_Ratio', '16_Birthday_Pattern']:
                    if 'Total' not in key and 'Digits' not in key:
                        pattern_name = key.replace('_', ' ')
                        pattern_list.append(pattern_name)
            
            top_patterns = pattern_list[:3]
            pattern_str = ", ".join(top_patterns) if top_patterns else "Various"
            
            total = sum(int(d) for d in num)
            print(f"{i:2d}. {num} (Score: {score:2d}) - {pattern_str} | Total: {total:2d}")
        
        return all_predictions
    
    def save_predictions_report(self, predictions: Dict):
        """Save complete predictions report to file"""
        filename = f"toto_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("REPORT MALAYSIA - 4D [TOTO] SPORTSTOTO / [88] SABAH 88 \n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Based on Data: {len(self.draw_dates)} times vote\n")
            f.write(f"Total Analyzed Numbers: {len(self.numbers_4d)}\n")
            if self.draw_dates and len(self.draw_dates) > 0:
                f.write(f"Julat Tarikh: {self.draw_dates[0].strftime('%Y-%m-%d')} until "
                       f"{self.draw_dates[-1].strftime('%Y-%m-%d')}\n\n")
            
            f.write("ANALYSIS OF 40 PATTERNS WITH 2 PREDICTIONS EACH:\n")
            f.write("-"*80 + "\n\n")
            
            for pattern_name, preds in predictions.items():
                f.write(f"{pattern_name}:\n")
                for i, pred in enumerate(preds, 1):
                    analysis = self.analyze_all_patterns_for_number(pred)
                    total = sum(int(d) for d in pred)
                    f.write(f"  {i}. {pred} (Total: {total})\n")
                f.write("\n")
            
            # Statistics
            f.write("\n" + "="*80 + "\n")
            f.write("IMPORTANT STATISTICS:\n")
            f.write("-"*80 + "\n\n")
            
            hc = self.get_hot_cold_digits()
            f.write("Frekuensi Digit (0-9):\n")
            for digit in range(10):
                freq = hc['all'].get(str(digit), 0)
                percentage = (freq / len(self.numbers_4d) * 100) if len(self.numbers_4d) > 0 else 0
                f.write(f"  {digit}: {freq} times ({percentage:.1f}%)\n")
            
            f.write(f"\nDigit HOT: {', '.join(hc['hot']) if hc['hot'] else 'None'}\n")
            f.write(f"Digit COLD: {', '.join(hc['cold']) if hc['cold'] else 'None'}\n")
            
            # Pattern frequency
            f.write("\nPattern Frequency (100 prime numbers):\n")
            pattern_counts = defaultdict(int)
            sample_size = min(100, len(self.numbers_4d))
            for num in self.numbers_4d[:sample_size]:
                analysis = self.analyze_all_patterns_for_number(num)
                for key, value in analysis.items():
                    if value and key not in ['13_Aritmatika_Difference', '14_Geometri_Ratio', '16_Birthday_Pattern']:
                        pattern_name = key.replace('_', ' ')
                        pattern_counts[pattern_name] += 1
            
            for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                percentage = (count / sample_size * 100)
                f.write(f"  {pattern}: {count} times ({percentage:.1f}%)\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("IMPORTANT: This prediction is based on statistical analysis only.\n")
            f.write("No guarantee of victory. Play responsibly.\n")
            f.write("="*80 + "\n")
        
        print(f"\n✅ The prediction report is saved as: {filename}")

def main():
    """Main execution function"""
    print("🎯 MALAYSIA - 4D [TOTO] SPORTSTOTO / [88] SABAH 88 - ALL ANALYSES WITH PREDICTIONS")
    print("="*60)
    

    if len(sys.argv) < 2:
        print("Usage: python3 toto_predictior2.py data.txt")
        sys.exit(1)

    predictor = TOTOPredictor40Analisis(sys.argv[1])
    # Gantikan 'toto_data.txt' dengan path file data anda
    #predictor = TOTOPredictor40Analisis('real_data.txt')
    
    if predictor.numbers_4d:
        # Generate semua prediksi
        predictions = predictor.generate_all_predictions()
        
        # Save report
        predictor.save_predictions_report(predictions)
        
        print("\n" + "="*60)
        print("ATTENTION:")
        print("="*60)
        print("1. This is purely a statistical analysis")
        print("2. None guaranteed to Win")
        print("3. Play responsibly")
        print("4. The actual results depend on luck")
        print("5. Anything is your own responsibility")
        print("="*60)
    else:
        print("❌ None data to analyze. Please make sure the data file exists and is in the correct format.")

if __name__ == "__main__":
    main()
